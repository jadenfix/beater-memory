use std::{net::SocketAddr, path::PathBuf};

use anyhow::Context;
use beater_memory::{
    BeaterJsJournal, EvalOptions, EvalSuite, LedgerEvent, MaintenanceOptions, MemoryEngine,
    MemoryMode, MemoryNodeKind, MemoryQuery, MemoryScope, MemoryServerConfig, ProjectReport,
    ReconstructionMode, ReconstructionOptions, SqliteMemoryStore, import_canonical_jsonl,
    run_eval_suite, serve,
};
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(name = "beater-memory")]
#[command(about = "Agent-first memory engine for Beater")]
struct Cli {
    /// Path to the beater-memory SQLite database.
    #[arg(long, global = true, default_value = ".beater-memory/memory.db")]
    db: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Create or migrate the memory database.
    Init,
    /// Import `beater.js` `.beater/journal.db`.
    ImportBeaterJs {
        #[arg(long)]
        journal: PathBuf,
        #[arg(long, default_value = "local")]
        tenant: String,
        #[arg(long, default_value = "default")]
        project: String,
        #[arg(long)]
        environment: Option<String>,
        #[arg(long, default_value_t = true)]
        project_pending: bool,
    },
    /// Import newline-delimited canonical span JSON.
    ImportJsonl {
        #[arg(long)]
        path: PathBuf,
        #[arg(long)]
        tenant: Option<String>,
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        environment: Option<String>,
        #[arg(long, default_value_t = true)]
        project_pending: bool,
    },
    /// Append a direct memory write and project it.
    Remember {
        #[arg(long, default_value = "local")]
        tenant: String,
        #[arg(long, default_value = "default")]
        project: String,
        #[arg(long)]
        environment: Option<String>,
        #[arg(long, value_enum, default_value_t = NodeKindArg::Fact)]
        kind: NodeKindArg,
        #[arg(long)]
        idempotency_key: Option<String>,
        #[arg(long)]
        no_project: bool,
        text: String,
    },
    /// Project pending ledger events into graph memory.
    Project {
        #[arg(long, default_value_t = 1000)]
        limit: usize,
    },
    /// Manage pending ledger events into graph memory.
    Manage {
        #[arg(long, default_value_t = 1000)]
        limit: usize,
    },
    /// Clear derived projections and replay the append-only ledger.
    RebuildProjection {
        #[arg(long)]
        yes_clear_projections: bool,
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
        #[arg(long)]
        max_events: Option<usize>,
    },
    /// Run integrity, schema, and count checks.
    Health {
        #[arg(long)]
        json: bool,
    },
    /// Run SQLite maintenance.
    Maintenance {
        #[arg(long)]
        vacuum: bool,
        #[arg(long)]
        repair_orphans: bool,
        /// Remove audit events older than this Unix millisecond timestamp.
        #[arg(long)]
        prune_audit_before_unix_ms: Option<i64>,
        /// Keep only the newest N audit events after any timestamp pruning.
        #[arg(long)]
        retain_audit_events: Option<usize>,
    },
    /// Create an online SQLite backup of the memory database.
    Backup {
        #[arg(long)]
        path: PathBuf,
    },
    /// Restore the active memory database from a SQLite backup.
    Restore {
        #[arg(long)]
        path: PathBuf,
        #[arg(long)]
        yes_replace_current_db: bool,
    },
    /// Run the authenticated HTTP API.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8765")]
        bind: SocketAddr,
        #[arg(long)]
        bearer_token: Option<String>,
        #[arg(long, default_value = "BEATER_MEMORY_TOKEN")]
        bearer_token_env: String,
        #[arg(long)]
        allow_no_auth: bool,
        #[arg(long, default_value_t = 1024 * 1024)]
        max_body_bytes: usize,
        #[arg(long, default_value_t = 10_000)]
        max_project_limit: usize,
        #[arg(long, default_value_t = 8_000)]
        max_query_tokens: u32,
        #[arg(long, default_value_t = 500)]
        max_audit_limit: usize,
        /// Maximum authenticated `/v1/*` requests per actor per minute. Set 0 to disable.
        #[arg(long, default_value_t = 600)]
        max_requests_per_minute: u32,
        /// Maximum concurrent blocking SQLite tasks. Set 0 to reject DB-backed requests.
        #[arg(long, default_value_t = 32)]
        max_concurrent_db_tasks: usize,
        /// Maximum wall time for one DB-backed HTTP task in milliseconds.
        #[arg(long, default_value_t = 30_000)]
        db_task_timeout_ms: u64,
    },
    /// Query memory and return an answer-shaped context bundle.
    Query {
        #[arg(long, default_value = "local")]
        tenant: String,
        #[arg(long, default_value = "default")]
        project: String,
        #[arg(long)]
        environment: Option<String>,
        #[arg(long, default_value_t = 1200)]
        max_tokens: u32,
        #[arg(long)]
        fresh: bool,
        #[arg(long)]
        as_of_unix_ms: Option<i64>,
        #[arg(long, value_enum, default_value_t = ReconstructionModeArg::Off)]
        reconstruction_mode: ReconstructionModeArg,
        #[arg(long, default_value_t = 4)]
        max_reconstruction_steps: u8,
        #[arg(long, default_value_t = 2_000)]
        max_reconstruction_tokens: u32,
        #[arg(long, value_delimiter = ',')]
        modes: Vec<ModeArg>,
        #[arg(long)]
        json: bool,
        question: String,
    },
    /// Print database counts.
    Stats,
    /// Run a deterministic in-memory evaluation suite.
    Eval {
        #[arg(long)]
        suite: PathBuf,
        #[arg(long)]
        max_tokens: Option<u32>,
        #[arg(long, value_enum)]
        reconstruction_mode: Option<ReconstructionModeArg>,
        #[arg(long)]
        max_reconstruction_steps: Option<u8>,
        #[arg(long)]
        max_reconstruction_tokens: Option<u32>,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum NodeKindArg {
    Episode,
    Fact,
    Procedure,
    State,
    Gotcha,
    AntiMemory,
    Topic,
}

impl From<NodeKindArg> for MemoryNodeKind {
    fn from(value: NodeKindArg) -> Self {
        match value {
            NodeKindArg::Episode => Self::Episode,
            NodeKindArg::Fact => Self::Fact,
            NodeKindArg::Procedure => Self::Procedure,
            NodeKindArg::State => Self::State,
            NodeKindArg::Gotcha => Self::Gotcha,
            NodeKindArg::AntiMemory => Self::AntiMemory,
            NodeKindArg::Topic => Self::Topic,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModeArg {
    Semantic,
    Episodic,
    Procedural,
    Gotcha,
    State,
}

impl From<ModeArg> for MemoryMode {
    fn from(value: ModeArg) -> Self {
        match value {
            ModeArg::Semantic => Self::Semantic,
            ModeArg::Episodic => Self::Episodic,
            ModeArg::Procedural => Self::Procedural,
            ModeArg::Gotcha => Self::Gotcha,
            ModeArg::State => Self::State,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ReconstructionModeArg {
    Off,
    Auto,
    Force,
}

impl From<ReconstructionModeArg> for ReconstructionMode {
    fn from(value: ReconstructionModeArg) -> Self {
        match value {
            ReconstructionModeArg::Off => Self::Off,
            ReconstructionModeArg::Auto => Self::Auto,
            ReconstructionModeArg::Force => Self::Force,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Init => {
            let engine = MemoryEngine::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&engine.store().stats()?)?
            );
        }
        Command::ImportBeaterJs {
            journal,
            tenant,
            project,
            environment,
            project_pending,
        } => {
            let engine = MemoryEngine::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            let report = BeaterJsJournal::new(journal).import_into(
                engine.store(),
                &tenant,
                &project,
                environment.as_deref(),
            )?;
            let project_report = if project_pending {
                Some(engine.project_pending(10_000)?)
            } else {
                None
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "import": {
                        "rows_seen": report.rows_seen,
                        "events_inserted": report.events_inserted,
                        "events_duplicate": report.events_duplicate,
                    },
                    "project": project_report,
                }))?
            );
        }
        Command::ImportJsonl {
            path,
            tenant,
            project,
            environment,
            project_pending,
        } => {
            let engine = MemoryEngine::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            let report = import_canonical_jsonl(
                path,
                engine.store(),
                tenant.as_deref(),
                project.as_deref(),
                environment.as_deref(),
            )?;
            let project_report = if project_pending {
                Some(engine.project_pending(10_000)?)
            } else {
                None
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "import": {
                        "rows_seen": report.rows_seen,
                        "events_inserted": report.events_inserted,
                        "events_duplicate": report.events_duplicate,
                    },
                    "project": project_report,
                }))?
            );
        }
        Command::Remember {
            tenant,
            project,
            environment,
            kind,
            idempotency_key,
            no_project,
            text,
        } => {
            let engine = MemoryEngine::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            if idempotency_key
                .as_deref()
                .is_some_and(|key| key.trim().is_empty())
            {
                anyhow::bail!("--idempotency-key must not be empty");
            }
            let mut event = LedgerEvent::direct_memory_write(
                &tenant,
                &project,
                MemoryNodeKind::from(kind),
                text,
            );
            event.environment_id = environment;
            if let Some(idempotency_key) = idempotency_key.as_deref() {
                event.apply_idempotency_key(idempotency_key);
            }
            engine.ingest_event(&event)?;
            let report = if no_project {
                ProjectReport::default()
            } else {
                engine.manage_pending(100)?
            };
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Project { limit } => {
            let engine = MemoryEngine::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            let report = engine.manage_pending(limit)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Manage { limit } => {
            let engine = MemoryEngine::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            let report = engine.manage_pending(limit)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::RebuildProjection {
            yes_clear_projections,
            batch_size,
            max_events,
        } => {
            if !yes_clear_projections {
                anyhow::bail!(
                    "rebuild clears derived projections; pass --yes-clear-projections to continue"
                );
            }
            let engine = MemoryEngine::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            let report = engine.rebuild_projection(batch_size, max_events)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Health { json } => {
            let engine = MemoryEngine::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            let health = engine.store().health()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&health)?);
            } else {
                println!(
                    "application_id: {}/{}",
                    health.application_id, health.expected_application_id
                );
                println!(
                    "schema: {}/{}",
                    health.schema_version, health.expected_schema_version
                );
                println!("integrity_ok: {}", health.integrity_ok);
                println!("foreign_key_violations: {}", health.foreign_key_violations);
                println!("graph_integrity_ok: {}", health.graph_integrity_ok);
                println!(
                    "graph_orphans: edges_from={} edges_to={} node_spans={} cue_index={}",
                    health.graph_integrity.orphan_edges_from,
                    health.graph_integrity.orphan_edges_to,
                    health.graph_integrity.orphan_node_spans,
                    health.graph_integrity.orphan_cue_index_entries
                );
                println!("ledger_events: {}", health.stats.ledger_events);
                println!("pending_events: {}", health.stats.pending_events);
                println!("nodes: {}", health.stats.nodes);
                println!("edges: {}", health.stats.edges);
                if !health.integrity_ok {
                    println!(
                        "integrity_messages: {}",
                        health.integrity_messages.join("; ")
                    );
                }
            }
        }
        Command::Maintenance {
            vacuum,
            repair_orphans,
            prune_audit_before_unix_ms,
            retain_audit_events,
        } => {
            let engine = MemoryEngine::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            let report = engine
                .store()
                .maintenance_with_options(MaintenanceOptions {
                    vacuum,
                    repair_orphans,
                    prune_audit_before_unix_ms,
                    retain_latest_audit_events: retain_audit_events,
                })?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Backup { path } => {
            let engine = MemoryEngine::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            let report = engine.store().backup_to(path)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Restore {
            path,
            yes_replace_current_db,
        } => {
            if !yes_replace_current_db {
                anyhow::bail!(
                    "restore replaces the active database; pass --yes-replace-current-db to continue"
                );
            }
            let mut store = SqliteMemoryStore::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            let report = store.restore_from(path)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Serve {
            bind,
            bearer_token,
            bearer_token_env,
            allow_no_auth,
            max_body_bytes,
            max_project_limit,
            max_query_tokens,
            max_audit_limit,
            max_requests_per_minute,
            max_concurrent_db_tasks,
            db_task_timeout_ms,
        } => {
            let token = bearer_token
                .or_else(|| std::env::var(&bearer_token_env).ok())
                .map(|token| token.trim().to_string())
                .filter(|token| !token.is_empty());
            if token.is_none() && !allow_no_auth {
                anyhow::bail!(
                    "refusing to start without auth; set --bearer-token, set {bearer_token_env}, or pass --allow-no-auth"
                );
            }
            let mut config = MemoryServerConfig::new(&cli.db, bind)
                .with_limits(max_body_bytes, max_project_limit, max_query_tokens)
                .with_audit_limit(max_audit_limit)
                .with_rate_limit(max_requests_per_minute)
                .with_db_concurrency_limit(max_concurrent_db_tasks)
                .with_db_task_timeout_ms(db_task_timeout_ms);
            if let Some(token) = token {
                config = config.with_bearer_token(token);
            }
            println!("serving beater-memory on http://{bind}");
            serve(config).await?;
        }
        Command::Query {
            tenant,
            project,
            environment,
            max_tokens,
            fresh,
            as_of_unix_ms,
            reconstruction_mode,
            max_reconstruction_steps,
            max_reconstruction_tokens,
            modes,
            json,
            question,
        } => {
            let engine = MemoryEngine::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            let mut scope = MemoryScope::new(tenant, project);
            if let Some(environment) = environment {
                scope = scope.with_environment(environment);
            }
            if let Some(as_of_unix_ms) = as_of_unix_ms {
                scope = scope.as_of_unix_ms(as_of_unix_ms);
            }
            let mut query = MemoryQuery::new(question, scope)
                .with_max_tokens(max_tokens)
                .with_reconstruction(ReconstructionOptions {
                    mode: reconstruction_mode.into(),
                    max_steps: max_reconstruction_steps,
                    max_tokens: max_reconstruction_tokens,
                });
            if fresh {
                query = query.requiring_fresh();
            }
            if !modes.is_empty() {
                query = query.with_modes(modes.into_iter().map(Into::into).collect());
            }
            let answer = engine.query(&query)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&answer)?);
            } else {
                println!("{}", answer.answer);
                println!();
                println!("tokens: {}", answer.token_estimate);
                println!("tier: {:?}", answer.tier_used);
                if let Some(routing) = answer.routing.as_ref() {
                    let routed_modes = serde_json::to_string(&routing.routed_modes)?;
                    let reason = serde_json::to_string(&routing.reason)?;
                    println!("routing: {routed_modes} via {}", reason.trim_matches('"'));
                    if let Some(reconstruction_modes) = routing.reconstruction_modes.as_ref() {
                        let reconstruction_modes = serde_json::to_string(reconstruction_modes)?;
                        println!("reconstruction routing: {reconstruction_modes}");
                    }
                }
                println!("evidence: {}", answer.evidence.len());
                if !answer.contradictions.is_empty() {
                    println!("contradictions: {}", answer.contradictions.len());
                }
                if !answer.stale_assumptions.is_empty() {
                    println!("stale_assumptions: {}", answer.stale_assumptions.len());
                }
            }
        }
        Command::Stats => {
            let engine = MemoryEngine::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&engine.store().stats()?)?
            );
        }
        Command::Eval {
            suite,
            max_tokens,
            reconstruction_mode,
            max_reconstruction_steps,
            max_reconstruction_tokens,
        } => {
            let suite_file =
                std::fs::File::open(&suite).with_context(|| format!("open {}", suite.display()))?;
            let suite_definition: EvalSuite = serde_json::from_reader(suite_file)
                .with_context(|| format!("parse {}", suite.display()))?;
            let report = run_eval_suite(
                &suite_definition,
                &EvalOptions {
                    max_tokens,
                    reconstruction_mode: reconstruction_mode.map(Into::into),
                    max_reconstruction_steps,
                    max_reconstruction_tokens,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if report.failed > 0 {
                anyhow::bail!(
                    "eval suite {} failed {}/{} case(s)",
                    report.suite,
                    report.failed,
                    report.cases
                );
            }
        }
    }
    Ok(())
}
