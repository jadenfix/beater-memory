use crate::{
    error::{MemoryError, MemoryResult},
    model::{BeliefRevisionOp, DistilledMemory, MemoryNodeKind, estimate_tokens},
    store::{LedgerEvent, MemoryNode},
    text::{concise, overlap_score},
};
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::{
    io::Write,
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const DEFAULT_COMMAND_PROVIDER_TIMEOUT_MS: u64 = 25_000;

/// Offline/sleep-time distillation boundary.
///
/// Provider-backed implementations should return only typed, validated output.
/// The engine revalidates this boundary before applying projection writes.
pub trait Distiller {
    fn distill(
        &self,
        event: &LedgerEvent,
        neighbors: &[MemoryNode],
    ) -> MemoryResult<DistillOutcome>;

    fn supports_late_replay(&self) -> bool {
        false
    }

    fn supports_projection_rebuild(&self) -> bool {
        false
    }
}

/// Provider/economics counters from one distillation attempt.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistillMetrics {
    pub provider_calls: usize,
    pub provider_errors: usize,
    pub schema_errors: usize,
    pub repair_attempts: usize,
    pub repair_successes: usize,
    pub rejected_outputs: usize,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub elapsed_ms: u64,
}

/// Distilled memory bundle plus counters. Rejected outcomes are not projected.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistillOutcome {
    pub memories: Vec<DistilledMemory>,
    pub metrics: DistillMetrics,
    pub rejected: bool,
}

impl DistillOutcome {
    #[must_use]
    pub fn accepted(memories: Vec<DistilledMemory>) -> Self {
        Self {
            memories,
            metrics: DistillMetrics::default(),
            rejected: false,
        }
    }

    #[must_use]
    pub fn rejected(metrics: DistillMetrics) -> Self {
        Self {
            memories: Vec::new(),
            metrics,
            rejected: true,
        }
    }
}

/// Synchronous provider interface for constrained distillation JSON.
pub trait DistillationProvider {
    fn distill(&self, prompt: DistillationPrompt<'_>) -> MemoryResult<String>;

    fn repair(&self, _prompt: DistillationRepairPrompt<'_>) -> MemoryResult<String> {
        Err(MemoryError::invalid(
            "distillation provider does not support repair",
        ))
    }
}

/// Input passed to a provider-backed distiller.
#[derive(Clone, Copy, Debug)]
pub struct DistillationPrompt<'a> {
    pub event: &'a LedgerEvent,
    pub neighbors: &'a [MemoryNode],
}

/// Repair request for malformed provider output.
#[derive(Clone, Copy, Debug)]
pub struct DistillationRepairPrompt<'a> {
    pub event: &'a LedgerEvent,
    pub neighbors: &'a [MemoryNode],
    pub raw_output: &'a str,
    pub error: &'a str,
}

/// Runtime distiller selection used by CLI/server entrypoints.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DistillerConfig {
    #[default]
    Heuristic,
    Command(CommandDistillationProviderConfig),
}

impl DistillerConfig {
    #[must_use]
    pub fn command(command: impl Into<PathBuf>) -> Self {
        Self::Command(CommandDistillationProviderConfig::new(command))
    }

    pub fn validate(&self) -> MemoryResult<()> {
        match self {
            Self::Heuristic => Ok(()),
            Self::Command(config) => config.validate(),
        }
    }

    pub fn build(&self) -> MemoryResult<RuntimeDistiller> {
        self.validate()?;
        Ok(match self {
            Self::Heuristic => RuntimeDistiller::Heuristic(HeuristicDistiller::default()),
            Self::Command(config) => RuntimeDistiller::Command(
                ProviderDistiller::new(CommandDistillationProvider::new(config.clone()))
                    .with_max_repairs(config.max_repairs),
            ),
        })
    }
}

/// Command provider configuration.
///
/// The command receives a JSON request on stdin and must write provider JSON on
/// stdout. The same command is used for repair requests with `request:
/// "repair"` and the original malformed output/error included.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandDistillationProviderConfig {
    pub command: PathBuf,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default = "default_command_provider_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_command_provider_max_repairs")]
    pub max_repairs: usize,
}

impl CommandDistillationProviderConfig {
    #[must_use]
    pub fn new(command: impl Into<PathBuf>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            timeout_ms: DEFAULT_COMMAND_PROVIDER_TIMEOUT_MS,
            max_repairs: 1,
        }
    }

    #[must_use]
    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    #[must_use]
    pub fn with_args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    #[must_use]
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    #[must_use]
    pub fn with_max_repairs(mut self, max_repairs: usize) -> Self {
        self.max_repairs = max_repairs;
        self
    }

    pub fn validate(&self) -> MemoryResult<()> {
        if self.command.as_os_str().is_empty() {
            return Err(MemoryError::invalid(
                "distillation command must not be empty",
            ));
        }
        if self.timeout_ms == 0 {
            return Err(MemoryError::invalid(
                "distillation command timeout_ms must be greater than 0",
            ));
        }
        Ok(())
    }
}

fn default_command_provider_timeout_ms() -> u64 {
    DEFAULT_COMMAND_PROVIDER_TIMEOUT_MS
}

fn default_command_provider_max_repairs() -> usize {
    1
}

/// Runtime distiller used when user-facing entrypoints select a distiller.
#[derive(Clone, Debug)]
pub enum RuntimeDistiller {
    Heuristic(HeuristicDistiller),
    Command(ProviderDistiller<CommandDistillationProvider>),
}

impl Distiller for RuntimeDistiller {
    fn distill(
        &self,
        event: &LedgerEvent,
        neighbors: &[MemoryNode],
    ) -> MemoryResult<DistillOutcome> {
        match self {
            Self::Heuristic(distiller) => distiller.distill(event, neighbors),
            Self::Command(distiller) => distiller.distill(event, neighbors),
        }
    }

    fn supports_late_replay(&self) -> bool {
        match self {
            Self::Heuristic(distiller) => distiller.supports_late_replay(),
            Self::Command(distiller) => distiller.supports_late_replay(),
        }
    }

    fn supports_projection_rebuild(&self) -> bool {
        match self {
            Self::Heuristic(distiller) => distiller.supports_projection_rebuild(),
            Self::Command(distiller) => distiller.supports_projection_rebuild(),
        }
    }
}

/// Provider adapter that delegates distillation JSON generation to a command.
#[derive(Clone, Debug)]
pub struct CommandDistillationProvider {
    config: CommandDistillationProviderConfig,
}

impl CommandDistillationProvider {
    #[must_use]
    pub fn new(config: CommandDistillationProviderConfig) -> Self {
        Self { config }
    }

    fn run(&self, request: CommandDistillationRequest<'_>) -> MemoryResult<String> {
        self.config.validate()?;
        let input = serde_json::to_vec(&request)?;
        let mut command = Command::new(&self.config.command);
        command
            .args(&self.config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        command.process_group(0);
        let mut child = command.spawn().map_err(|err| {
            MemoryError::invalid(format!(
                "failed to start distillation command {:?}: {err}",
                self.config.command
            ))
        })?;
        let Some(mut stdin) = child.stdin.take() else {
            terminate_child(&mut child);
            let _ = child.wait();
            return Err(MemoryError::invalid(
                "distillation command stdin was unavailable",
            ));
        };

        let deadline = Instant::now() + Duration::from_millis(self.config.timeout_ms);
        let writer = thread::spawn(move || -> MemoryResult<()> {
            stdin.write_all(&input)?;
            drop(stdin);
            Ok(())
        });
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let output = child.wait_with_output()?;
                    let write_result = writer.join().map_err(|_| {
                        MemoryError::invalid("distillation command stdin writer panicked")
                    })?;
                    if !status.success() {
                        return Err(MemoryError::invalid(format!(
                            "distillation command exited with status {status}: {}",
                            String::from_utf8_lossy(&output.stderr)
                        )));
                    }
                    write_result?;
                    return String::from_utf8(output.stdout).map_err(|err| {
                        MemoryError::invalid(format!("provider output is not UTF-8: {err}"))
                    });
                }
                Ok(None) => {}
                Err(err) => {
                    terminate_child(&mut child);
                    let _ = child.wait();
                    let _ = writer.join();
                    return Err(err.into());
                }
            }
            if Instant::now() >= deadline {
                terminate_child(&mut child);
                let _ = child.wait();
                let _ = writer.join();
                return Err(MemoryError::invalid(format!(
                    "distillation command exceeded timeout_ms {}",
                    self.config.timeout_ms
                )));
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
}

fn terminate_child(child: &mut Child) {
    #[cfg(unix)]
    {
        let pgid = child.id() as libc::pid_t;
        unsafe {
            libc::kill(-pgid, libc::SIGKILL);
        }
    }
    let _ = child.kill();
}

impl DistillationProvider for CommandDistillationProvider {
    fn distill(&self, prompt: DistillationPrompt<'_>) -> MemoryResult<String> {
        self.run(CommandDistillationRequest {
            request: "distill",
            event: prompt.event,
            neighbors: prompt.neighbors,
            raw_output: None,
            error: None,
        })
    }

    fn repair(&self, prompt: DistillationRepairPrompt<'_>) -> MemoryResult<String> {
        self.run(CommandDistillationRequest {
            request: "repair",
            event: prompt.event,
            neighbors: prompt.neighbors,
            raw_output: Some(prompt.raw_output),
            error: Some(prompt.error),
        })
    }
}

#[derive(Serialize)]
struct CommandDistillationRequest<'a> {
    request: &'static str,
    event: &'a LedgerEvent,
    neighbors: &'a [MemoryNode],
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_output: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

/// Provider-backed distiller with schema validation and bounded repair.
#[derive(Clone, Debug)]
pub struct ProviderDistiller<P> {
    provider: P,
    max_repairs: usize,
}

impl<P> ProviderDistiller<P> {
    #[must_use]
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            max_repairs: 1,
        }
    }

    #[must_use]
    pub fn with_max_repairs(mut self, max_repairs: usize) -> Self {
        self.max_repairs = max_repairs;
        self
    }
}

/// Deterministic first-principles distiller used by the local MVP.
#[derive(Clone, Debug)]
pub struct HeuristicDistiller {
    max_memory_chars: usize,
}

impl Default for HeuristicDistiller {
    fn default() -> Self {
        Self {
            max_memory_chars: 900,
        }
    }
}

impl HeuristicDistiller {
    #[must_use]
    pub fn new(max_memory_chars: usize) -> Self {
        Self { max_memory_chars }
    }
}

impl Distiller for HeuristicDistiller {
    fn distill(
        &self,
        event: &LedgerEvent,
        neighbors: &[MemoryNode],
    ) -> MemoryResult<DistillOutcome> {
        let body = concise(&event.text, self.max_memory_chars);
        if body.trim().is_empty() {
            return Ok(DistillOutcome::accepted(vec![DistilledMemory {
                op: BeliefRevisionOp::Noop,
                node_kind: MemoryNodeKind::Episode,
                text: String::new(),
                target_node_id: None,
                cited_spans: vec![event.cited_span()],
            }]));
        }

        let cited_span = event.cited_span();
        let mut out = vec![DistilledMemory::add(
            MemoryNodeKind::Episode,
            format!("{} {}: {body}", event.span_kind, event.name),
            cited_span.clone(),
        )];

        let kind = classify_memory_kind(event, &body);
        let op = classify_op(&body);
        let target_node_id = match op {
            BeliefRevisionOp::Update => best_target(&body, kind, neighbors, 0.35),
            BeliefRevisionOp::Invalidate => best_target(&body, kind, neighbors, 0.12),
            BeliefRevisionOp::Add | BeliefRevisionOp::Noop => None,
        };

        out.push(DistilledMemory {
            op,
            node_kind: kind,
            text: body,
            target_node_id,
            cited_spans: vec![cited_span],
        });
        Ok(DistillOutcome::accepted(out))
    }

    fn supports_late_replay(&self) -> bool {
        true
    }

    fn supports_projection_rebuild(&self) -> bool {
        true
    }
}

impl<P: DistillationProvider> Distiller for ProviderDistiller<P> {
    fn distill(
        &self,
        event: &LedgerEvent,
        neighbors: &[MemoryNode],
    ) -> MemoryResult<DistillOutcome> {
        let started = Instant::now();
        let mut metrics = DistillMetrics {
            input_tokens: estimate_distillation_input_tokens(event, neighbors),
            ..DistillMetrics::default()
        };
        metrics.provider_calls += 1;
        let mut raw = match self
            .provider
            .distill(DistillationPrompt { event, neighbors })
        {
            Ok(raw) => raw,
            Err(_) => {
                metrics.provider_errors += 1;
                metrics.rejected_outputs += 1;
                metrics.elapsed_ms = elapsed_ms(started);
                return Ok(DistillOutcome::rejected(metrics));
            }
        };
        metrics.output_tokens += estimate_tokens(&raw);

        let mut repairs_used = 0;
        loop {
            match parse_provider_output(&raw) {
                Ok(memories) => {
                    if repairs_used > 0 {
                        metrics.repair_successes += 1;
                    }
                    metrics.elapsed_ms = elapsed_ms(started);
                    return Ok(DistillOutcome {
                        memories,
                        metrics,
                        rejected: false,
                    });
                }
                Err(error) => {
                    metrics.schema_errors += 1;
                    if repairs_used >= self.max_repairs {
                        metrics.rejected_outputs += 1;
                        metrics.elapsed_ms = elapsed_ms(started);
                        return Ok(DistillOutcome::rejected(metrics));
                    }
                    metrics.repair_attempts += 1;
                    metrics.provider_calls += 1;
                    metrics.input_tokens += estimate_distillation_input_tokens(event, neighbors)
                        + estimate_tokens(&raw)
                        + estimate_tokens(&error);
                    let repaired = match self.provider.repair(DistillationRepairPrompt {
                        event,
                        neighbors,
                        raw_output: &raw,
                        error: &error,
                    }) {
                        Ok(repaired) => repaired,
                        Err(_) => {
                            metrics.provider_errors += 1;
                            metrics.rejected_outputs += 1;
                            metrics.elapsed_ms = elapsed_ms(started);
                            return Ok(DistillOutcome::rejected(metrics));
                        }
                    };
                    metrics.output_tokens += estimate_tokens(&repaired);
                    raw = repaired;
                    repairs_used += 1;
                }
            }
        }
    }

    fn supports_projection_rebuild(&self) -> bool {
        false
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderOutput {
    memories: Vec<ProviderMemory>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderMemory {
    op: BeliefRevisionOp,
    node_kind: MemoryNodeKind,
    text: String,
    target_node_id: Option<String>,
    cited_spans: Vec<crate::model::CitedSpan>,
}

fn parse_provider_output(raw: &str) -> Result<Vec<DistilledMemory>, String> {
    let output: ProviderOutput =
        serde_json::from_str(raw).map_err(|err| format!("invalid provider JSON: {err}"))?;
    if output.memories.is_empty() {
        return Err("provider output must include at least one memory".to_string());
    }
    let memories = output
        .memories
        .into_iter()
        .map(|memory| DistilledMemory {
            op: memory.op,
            node_kind: memory.node_kind,
            text: memory.text,
            target_node_id: memory.target_node_id,
            cited_spans: memory.cited_spans,
        })
        .collect::<Vec<_>>();
    for (index, memory) in memories.iter().enumerate() {
        memory
            .validate()
            .map_err(|err| format!("invalid memory at index {index}: {err}"))?;
    }
    Ok(memories)
}

fn estimate_distillation_input_tokens(event: &LedgerEvent, neighbors: &[MemoryNode]) -> u32 {
    estimate_tokens(&event.text)
        + neighbors
            .iter()
            .map(|neighbor| estimate_tokens(&neighbor.text))
            .sum::<u32>()
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

fn classify_memory_kind(event: &LedgerEvent, text: &str) -> MemoryNodeKind {
    let declared = event.name.to_ascii_lowercase();
    match declared.as_str() {
        "fact" | "semantic" => return MemoryNodeKind::Fact,
        "episode" | "episodic" => return MemoryNodeKind::Episode,
        "procedure" | "runbook" | "workflow" => return MemoryNodeKind::Procedure,
        "state" => return MemoryNodeKind::State,
        "gotcha" | "failure" => return MemoryNodeKind::Gotcha,
        "anti_memory" | "anti-memory" => return MemoryNodeKind::AntiMemory,
        _ => {}
    }

    let lower = text.to_ascii_lowercase();
    if lower.contains("do not use")
        || lower.contains("looked relevant")
        || lower.contains("misleading")
        || lower.contains("red herring")
    {
        MemoryNodeKind::AntiMemory
    } else if lower.contains("error")
        || lower.contains("failed")
        || lower.contains("failure")
        || lower.contains("panic")
        || lower.contains("regression")
        || lower.contains("gotcha")
        || lower.contains("blocked")
    {
        MemoryNodeKind::Gotcha
    } else if lower.contains("run ")
        || lower.contains("command")
        || lower.contains("step")
        || lower.contains("fix by")
        || lower.contains("workaround")
        || lower.contains("procedure")
    {
        MemoryNodeKind::Procedure
    } else if lower.contains("current ")
        || lower.contains("configured")
        || lower.contains("environment")
        || lower.contains("state")
        || lower.contains("uses ")
    {
        MemoryNodeKind::State
    } else {
        MemoryNodeKind::Fact
    }
}

fn classify_op(text: &str) -> BeliefRevisionOp {
    let lower = text.to_ascii_lowercase();
    if lower.contains("no longer")
        || lower.contains("deprecated")
        || lower.contains("invalidated")
        || lower.contains("stale")
        || lower.contains("not true")
        || lower.contains("replace ")
        || lower.contains("instead of")
        || lower.contains("do not use")
    {
        BeliefRevisionOp::Invalidate
    } else if lower.contains("update")
        || lower.contains("changed")
        || lower.contains("now ")
        || lower.contains("new ")
    {
        BeliefRevisionOp::Update
    } else {
        BeliefRevisionOp::Add
    }
}

fn best_target(
    text: &str,
    kind: MemoryNodeKind,
    neighbors: &[MemoryNode],
    min_overlap: f32,
) -> Option<String> {
    neighbors
        .iter()
        .filter(|node| node.kind == kind)
        .filter(|node| {
            !matches!(
                node.kind,
                MemoryNodeKind::Episode | MemoryNodeKind::EntityCue
            )
        })
        .map(|node| (overlap_score(text, &node.text), node))
        .filter(|(score, _)| *score >= min_overlap)
        .max_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, node)| node.id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(kind: MemoryNodeKind, text: &str) -> LedgerEvent {
        LedgerEvent::direct_memory_write("tenant", "project", kind, text)
    }

    #[test]
    fn emits_episode_plus_typed_memory() {
        let memories = HeuristicDistiller::default()
            .distill(
                &event(
                    MemoryNodeKind::Gotcha,
                    "Checkout failed with DATABASE_URL missing. Fix by setting it.",
                ),
                &[],
            )
            .unwrap_or_else(|err| panic!("{err}"))
            .memories;

        assert_eq!(memories.len(), 2);
        assert_eq!(memories[0].node_kind, MemoryNodeKind::Episode);
        assert_eq!(memories[1].node_kind, MemoryNodeKind::Gotcha);
    }

    #[test]
    fn invalidations_can_target_neighbors() {
        let neighbor = MemoryNode {
            id: "node_old".to_string(),
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            kind: MemoryNodeKind::Fact,
            text: "Use the old checkout token.".to_string(),
            canonical_key: "fact:old checkout token".to_string(),
            created_at_unix_ms: 1,
            updated_at_unix_ms: 1,
            valid_from_unix_ms: 1,
            valid_to_unix_ms: None,
            valid_to_event_id: None,
            confidence: 0.7,
            token_estimate: 8,
            observation_count: 1,
        };
        let memories = HeuristicDistiller::default()
            .distill(
                &event(
                    MemoryNodeKind::Fact,
                    "Do not use the old checkout token; it is deprecated.",
                ),
                &[neighbor],
            )
            .unwrap_or_else(|err| panic!("{err}"))
            .memories;

        assert_eq!(memories[1].op, BeliefRevisionOp::Invalidate);
        assert_eq!(memories[1].target_node_id.as_deref(), Some("node_old"));
    }

    #[test]
    fn invalidations_prefer_typed_memory_over_episode_scaffolding() {
        let episode = MemoryNode {
            id: "node_episode".to_string(),
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            kind: MemoryNodeKind::Episode,
            text: "memory.write fact: Use the old checkout token.".to_string(),
            canonical_key: "episode:trace:span:1".to_string(),
            created_at_unix_ms: 1,
            updated_at_unix_ms: 1,
            valid_from_unix_ms: 1,
            valid_to_unix_ms: None,
            valid_to_event_id: None,
            confidence: 0.7,
            token_estimate: 8,
            observation_count: 1,
        };
        let fact = MemoryNode {
            id: "node_fact".to_string(),
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            kind: MemoryNodeKind::Fact,
            text: "Use the old checkout token.".to_string(),
            canonical_key: "fact:old checkout token".to_string(),
            created_at_unix_ms: 1,
            updated_at_unix_ms: 1,
            valid_from_unix_ms: 1,
            valid_to_unix_ms: None,
            valid_to_event_id: None,
            confidence: 0.7,
            token_estimate: 8,
            observation_count: 1,
        };

        let memories = HeuristicDistiller::default()
            .distill(
                &event(
                    MemoryNodeKind::Fact,
                    "Do not use the old checkout token; it is deprecated.",
                ),
                &[episode, fact],
            )
            .unwrap_or_else(|err| panic!("{err}"))
            .memories;

        assert_eq!(memories[1].op, BeliefRevisionOp::Invalidate);
        assert_eq!(memories[1].target_node_id.as_deref(), Some("node_fact"));
    }

    #[derive(Clone)]
    struct FakeProvider {
        raw: Result<String, String>,
        repaired: Option<String>,
    }

    impl DistillationProvider for FakeProvider {
        fn distill(&self, _prompt: DistillationPrompt<'_>) -> MemoryResult<String> {
            self.raw
                .clone()
                .map_err(|err| MemoryError::invalid(format!("provider failed: {err}")))
        }

        fn repair(&self, prompt: DistillationRepairPrompt<'_>) -> MemoryResult<String> {
            assert!(!prompt.raw_output.is_empty());
            assert!(!prompt.error.is_empty());
            self.repaired
                .clone()
                .ok_or_else(|| MemoryError::invalid("repair unavailable"))
        }
    }

    #[cfg(unix)]
    #[test]
    fn command_provider_timeout_covers_blocked_stdin() {
        let text = "x".repeat(2 * 1024 * 1024);
        let event = event(MemoryNodeKind::Fact, &text);
        let provider = CommandDistillationProvider::new(
            CommandDistillationProviderConfig::new("sh")
                .with_args(["-c", "sleep 2"])
                .with_timeout_ms(50),
        );
        let started = Instant::now();

        let err = provider
            .distill(DistillationPrompt {
                event: &event,
                neighbors: &[],
            })
            .unwrap_err();

        assert!(err.to_string().contains("timeout_ms"));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[cfg(unix)]
    #[test]
    fn command_provider_rejects_non_utf8_output() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let provider = CommandDistillationProvider::new(
            CommandDistillationProviderConfig::new("sh")
                .with_args(["-c", "cat >/dev/null; printf '\\377'"])
                .with_timeout_ms(1_000),
        );

        let err = provider
            .distill(DistillationPrompt {
                event: &event,
                neighbors: &[],
            })
            .unwrap_err();

        assert!(err.to_string().contains("UTF-8"));
    }

    fn provider_json(event: &LedgerEvent, text: &str) -> String {
        serde_json::json!({
            "memories": [{
                "op": "add",
                "node_kind": "fact",
                "text": text,
                "target_node_id": null,
                "cited_spans": [event.cited_span()]
            }]
        })
        .to_string()
    }

    #[test]
    fn provider_distiller_accepts_valid_schema() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let distiller = ProviderDistiller::new(FakeProvider {
            raw: Ok(provider_json(&event, "Checkout uses DATABASE_URL.")),
            repaired: None,
        });

        let outcome = distiller
            .distill(&event, &[])
            .unwrap_or_else(|err| panic!("{err}"));

        assert!(!outcome.rejected);
        assert_eq!(outcome.memories.len(), 1);
        assert_eq!(outcome.metrics.provider_calls, 1);
        assert_eq!(outcome.metrics.repair_attempts, 0);
        assert!(outcome.metrics.input_tokens > 0);
        assert!(outcome.metrics.output_tokens > 0);
    }

    #[test]
    fn provider_distiller_repairs_invalid_schema_once() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let distiller = ProviderDistiller::new(FakeProvider {
            raw: Ok("{\"memories\":".to_string()),
            repaired: Some(provider_json(&event, "Checkout uses DATABASE_URL.")),
        });

        let outcome = distiller
            .distill(&event, &[])
            .unwrap_or_else(|err| panic!("{err}"));

        assert!(!outcome.rejected);
        assert_eq!(outcome.metrics.provider_calls, 2);
        assert_eq!(outcome.metrics.schema_errors, 1);
        assert_eq!(outcome.metrics.repair_attempts, 1);
        assert_eq!(outcome.metrics.repair_successes, 1);
        assert!(outcome.metrics.input_tokens > estimate_distillation_input_tokens(&event, &[]));
    }

    #[test]
    fn provider_distiller_rejects_after_failed_repair() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let distiller = ProviderDistiller::new(FakeProvider {
            raw: Ok("{\"memories\":".to_string()),
            repaired: Some("{\"memories\":".to_string()),
        });

        let outcome = distiller
            .distill(&event, &[])
            .unwrap_or_else(|err| panic!("{err}"));

        assert!(outcome.rejected);
        assert!(outcome.memories.is_empty());
        assert_eq!(outcome.metrics.schema_errors, 2);
        assert_eq!(outcome.metrics.rejected_outputs, 1);
    }

    #[test]
    fn provider_distiller_rejects_transport_failure() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let distiller = ProviderDistiller::new(FakeProvider {
            raw: Err("timeout".to_string()),
            repaired: None,
        });

        let outcome = distiller
            .distill(&event, &[])
            .unwrap_or_else(|err| panic!("{err}"));

        assert!(outcome.rejected);
        assert_eq!(outcome.metrics.provider_errors, 1);
        assert_eq!(outcome.metrics.rejected_outputs, 1);
    }

    #[test]
    fn provider_distiller_rejects_unknown_fields() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let raw = serde_json::json!({
            "memories": [{
                "op": "add",
                "node_kind": "fact",
                "text": "Checkout uses DATABASE_URL.",
                "target_node_id": null,
                "cited_spans": [event.cited_span()],
                "unexpected": true
            }]
        })
        .to_string();
        let distiller = ProviderDistiller::new(FakeProvider {
            raw: Ok(raw),
            repaired: None,
        });

        let outcome = distiller
            .distill(&event, &[])
            .unwrap_or_else(|err| panic!("{err}"));

        assert!(outcome.rejected);
        assert_eq!(outcome.metrics.schema_errors, 1);
        assert_eq!(outcome.metrics.repair_attempts, 1);
        assert_eq!(outcome.metrics.rejected_outputs, 1);
    }

    #[test]
    fn provider_distiller_rejects_empty_batches() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let distiller = ProviderDistiller::new(FakeProvider {
            raw: Ok(serde_json::json!({ "memories": [] }).to_string()),
            repaired: None,
        });

        let outcome = distiller
            .distill(&event, &[])
            .unwrap_or_else(|err| panic!("{err}"));

        assert!(outcome.rejected);
        assert_eq!(outcome.metrics.schema_errors, 1);
        assert_eq!(outcome.metrics.rejected_outputs, 1);
    }

    #[derive(Clone)]
    struct TwoRepairProvider;

    impl DistillationProvider for TwoRepairProvider {
        fn distill(&self, _prompt: DistillationPrompt<'_>) -> MemoryResult<String> {
            Ok("{\"memories\":".to_string())
        }

        fn repair(&self, prompt: DistillationRepairPrompt<'_>) -> MemoryResult<String> {
            if prompt.raw_output == "{\"memories\":" {
                Ok("{\"memories\":[]}".to_string())
            } else {
                Ok(provider_json(prompt.event, "Checkout uses DATABASE_URL."))
            }
        }
    }

    #[test]
    fn provider_distiller_honors_max_repairs() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let distiller = ProviderDistiller::new(TwoRepairProvider).with_max_repairs(2);

        let outcome = distiller
            .distill(&event, &[])
            .unwrap_or_else(|err| panic!("{err}"));

        assert!(!outcome.rejected);
        assert_eq!(outcome.metrics.provider_calls, 3);
        assert_eq!(outcome.metrics.schema_errors, 2);
        assert_eq!(outcome.metrics.repair_attempts, 2);
        assert_eq!(outcome.metrics.repair_successes, 1);
    }
}
