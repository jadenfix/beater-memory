use std::path::PathBuf;

use anyhow::Context;
use beater_memory::{
    BeaterJsJournal, LedgerEvent, MemoryEngine, MemoryMode, MemoryNodeKind, MemoryQuery,
    MemoryScope, import_canonical_jsonl,
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
        text: String,
    },
    /// Project pending ledger events into graph memory.
    Project {
        #[arg(long, default_value_t = 1000)]
        limit: usize,
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
        #[arg(long, value_delimiter = ',')]
        modes: Vec<ModeArg>,
        #[arg(long)]
        json: bool,
        question: String,
    },
    /// Print database counts.
    Stats,
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

fn main() -> anyhow::Result<()> {
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
                    "project": project_report.map(|report| serde_json::json!({
                        "events_seen": report.events_seen,
                        "memories_added": report.memories_added,
                        "memories_updated": report.memories_updated,
                        "memories_invalidated": report.memories_invalidated,
                        "memories_nooped": report.memories_nooped,
                        "edges_added": report.edges_added,
                    })),
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
                    "project": project_report.map(|report| serde_json::json!({
                        "events_seen": report.events_seen,
                        "memories_added": report.memories_added,
                        "memories_updated": report.memories_updated,
                        "memories_invalidated": report.memories_invalidated,
                        "memories_nooped": report.memories_nooped,
                        "edges_added": report.edges_added,
                    })),
                }))?
            );
        }
        Command::Remember {
            tenant,
            project,
            environment,
            kind,
            text,
        } => {
            let engine = MemoryEngine::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            let mut event = LedgerEvent::direct_memory_write(
                &tenant,
                &project,
                MemoryNodeKind::from(kind),
                text,
            );
            event.environment_id = environment;
            engine.ingest_event(&event)?;
            let report = engine.project_pending(100)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "project": {
                        "events_seen": report.events_seen,
                        "memories_added": report.memories_added,
                        "memories_updated": report.memories_updated,
                        "memories_invalidated": report.memories_invalidated,
                        "memories_nooped": report.memories_nooped,
                        "edges_added": report.edges_added,
                    }
                }))?
            );
        }
        Command::Project { limit } => {
            let engine = MemoryEngine::open(&cli.db)
                .with_context(|| format!("open {}", cli.db.display()))?;
            let report = engine.project_pending(limit)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "events_seen": report.events_seen,
                    "memories_added": report.memories_added,
                    "memories_updated": report.memories_updated,
                    "memories_invalidated": report.memories_invalidated,
                    "memories_nooped": report.memories_nooped,
                    "edges_added": report.edges_added,
                }))?
            );
        }
        Command::Query {
            tenant,
            project,
            environment,
            max_tokens,
            fresh,
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
            let mut query = MemoryQuery::new(question, scope).with_max_tokens(max_tokens);
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
    }
    Ok(())
}
