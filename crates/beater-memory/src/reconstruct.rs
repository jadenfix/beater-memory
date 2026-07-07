use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::os::unix::{fs::PermissionsExt, process::CommandExt};
use std::{
    env, fs,
    fs::{File, OpenOptions},
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    error::{MemoryError, MemoryResult},
    model::{MemoryNodeKind, estimate_tokens},
};

const DEFAULT_COMMAND_RECONSTRUCTOR_TIMEOUT_MS: u64 = 25_000;

/// Candidate visible to a read-time active reconstruction policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionCandidate {
    pub node_id: String,
    pub kind: MemoryNodeKind,
    pub text: String,
    pub score: f32,
    pub token_estimate: u32,
}

/// One bounded reconstruction step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionStep {
    pub question: String,
    pub step_index: u8,
    pub expanded_node_id: String,
    pub remaining_tokens: u32,
    pub candidates: Vec<ReconstructionCandidate>,
}

/// Validated action returned by an active reconstruction policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum ReconstructionDecision {
    Accept { node_id: String },
    Prune { node_id: String },
    Stop,
}

impl ReconstructionDecision {
    #[must_use]
    pub fn into_outcome(self) -> ReconstructionDecisionOutcome {
        ReconstructionDecisionOutcome {
            decision: self,
            metrics: ReconstructionMetrics::default(),
        }
    }
}

/// Provider counters from one active reconstruction decision.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconstructionMetrics {
    pub provider_calls: usize,
    pub provider_errors: usize,
    pub schema_errors: usize,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub elapsed_ms: u64,
}

/// Validated reconstruction decision plus provider/economics counters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconstructionDecisionOutcome {
    pub decision: ReconstructionDecision,
    pub metrics: ReconstructionMetrics,
}

/// Provider-neutral hook for Tier 2 read-time graph exploration.
pub trait ActiveReconstructor: Clone + Send + Sync + 'static {
    fn decide(&self, step: &ReconstructionStep) -> MemoryResult<ReconstructionDecisionOutcome>;
}

/// Deterministic, token-free reconstruction policy used by the local engine.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeterministicReconstructor;

impl ActiveReconstructor for DeterministicReconstructor {
    fn decide(&self, step: &ReconstructionStep) -> MemoryResult<ReconstructionDecisionOutcome> {
        Ok(step
            .candidates
            .iter()
            .filter(|candidate| candidate.token_estimate <= step.remaining_tokens)
            .max_by(|left, right| left.score.total_cmp(&right.score))
            .map(|candidate| ReconstructionDecision::Accept {
                node_id: candidate.node_id.clone(),
            })
            .unwrap_or(ReconstructionDecision::Stop)
            .into_outcome())
    }
}

/// Runtime active reconstruction selection used by CLI/server entrypoints.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReconstructorConfig {
    #[default]
    Deterministic,
    Command(CommandReconstructionProviderConfig),
}

impl ReconstructorConfig {
    #[must_use]
    pub fn command(command: impl Into<PathBuf>) -> Self {
        Self::Command(CommandReconstructionProviderConfig::new(command))
    }

    pub fn validate(&self) -> MemoryResult<()> {
        match self {
            Self::Deterministic => Ok(()),
            Self::Command(config) => config.validate(),
        }
    }

    pub fn build(&self) -> MemoryResult<RuntimeReconstructor> {
        self.validate()?;
        Ok(match self {
            Self::Deterministic => RuntimeReconstructor::Deterministic(DeterministicReconstructor),
            Self::Command(config) => RuntimeReconstructor::Command(ProviderReconstructor::new(
                CommandReconstructionProvider::new(config.clone()),
            )),
        })
    }
}

/// Command provider configuration for read-time active reconstruction.
///
/// The command receives one reconstruction step as JSON on stdin and must write
/// a decision JSON object on stdout:
/// `{ "decision": "accept", "node_id": "node_..." }`,
/// `{ "decision": "prune", "node_id": "node_..." }`, or
/// `{ "decision": "stop" }`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandReconstructionProviderConfig {
    pub command: PathBuf,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default = "default_command_reconstructor_timeout_ms")]
    pub timeout_ms: u64,
}

impl CommandReconstructionProviderConfig {
    #[must_use]
    pub fn new(command: impl Into<PathBuf>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            timeout_ms: DEFAULT_COMMAND_RECONSTRUCTOR_TIMEOUT_MS,
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

    pub fn validate(&self) -> MemoryResult<()> {
        if self.command.as_os_str().is_empty() {
            return Err(MemoryError::invalid(
                "reconstruction command must not be empty",
            ));
        }
        if self.timeout_ms == 0 {
            return Err(MemoryError::invalid(
                "reconstruction command timeout_ms must be greater than 0",
            ));
        }
        validate_command_path(&self.command)?;
        Ok(())
    }
}

const fn default_command_reconstructor_timeout_ms() -> u64 {
    DEFAULT_COMMAND_RECONSTRUCTOR_TIMEOUT_MS
}

/// Runtime reconstructor built from `ReconstructorConfig`.
#[derive(Clone, Debug)]
pub enum RuntimeReconstructor {
    Deterministic(DeterministicReconstructor),
    Command(ProviderReconstructor<CommandReconstructionProvider>),
}

impl ActiveReconstructor for RuntimeReconstructor {
    fn decide(&self, step: &ReconstructionStep) -> MemoryResult<ReconstructionDecisionOutcome> {
        match self {
            Self::Deterministic(reconstructor) => reconstructor.decide(step),
            Self::Command(reconstructor) => reconstructor.decide(step),
        }
    }
}

/// Provider interface for constrained active-reconstruction decisions.
pub trait ReconstructionProvider: Clone + Send + Sync + 'static {
    fn decide(&self, step: &ReconstructionStep) -> MemoryResult<String>;
}

/// Provider-backed active reconstructor with schema validation.
#[derive(Clone, Debug)]
pub struct ProviderReconstructor<P> {
    provider: P,
}

impl<P> ProviderReconstructor<P> {
    #[must_use]
    pub fn new(provider: P) -> Self {
        Self { provider }
    }
}

impl<P: ReconstructionProvider> ActiveReconstructor for ProviderReconstructor<P> {
    fn decide(&self, step: &ReconstructionStep) -> MemoryResult<ReconstructionDecisionOutcome> {
        let started = Instant::now();
        let mut metrics = ReconstructionMetrics {
            provider_calls: 1,
            input_tokens: estimate_reconstruction_input_tokens(step),
            ..ReconstructionMetrics::default()
        };
        let raw = match self.provider.decide(step) {
            Ok(raw) => raw,
            Err(_) => {
                metrics.provider_errors += 1;
                metrics.elapsed_ms = elapsed_ms(started);
                return Ok(ReconstructionDecisionOutcome {
                    decision: ReconstructionDecision::Stop,
                    metrics,
                });
            }
        };
        metrics.output_tokens += estimate_tokens(&raw);
        match parse_provider_reconstruction_output(&raw, step) {
            Ok(decision) => {
                metrics.elapsed_ms = elapsed_ms(started);
                Ok(ReconstructionDecisionOutcome { decision, metrics })
            }
            Err(_) => {
                metrics.schema_errors += 1;
                metrics.elapsed_ms = elapsed_ms(started);
                Ok(ReconstructionDecisionOutcome {
                    decision: ReconstructionDecision::Stop,
                    metrics,
                })
            }
        }
    }
}

/// Command-backed reconstruction provider.
#[derive(Clone, Debug)]
pub struct CommandReconstructionProvider {
    config: CommandReconstructionProviderConfig,
}

impl CommandReconstructionProvider {
    #[must_use]
    pub fn new(config: CommandReconstructionProviderConfig) -> Self {
        Self { config }
    }
}

impl ReconstructionProvider for CommandReconstructionProvider {
    fn decide(&self, step: &ReconstructionStep) -> MemoryResult<String> {
        self.config.validate()?;
        let input = serde_json::to_vec(step)?;
        let output_files = CommandOutputFiles::create()?;
        let stdout_file = output_files.stdout_file()?;
        let stderr_file = output_files.stderr_file()?;
        let mut command = Command::new(&self.config.command);
        command
            .args(&self.config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::from(stdout_file))
            .stderr(Stdio::from(stderr_file));
        #[cfg(unix)]
        command.process_group(0);
        let mut child = command.spawn().map_err(|err| {
            MemoryError::invalid(format!(
                "failed to start reconstruction command {:?}: {err}",
                self.config.command
            ))
        })?;
        let Some(mut stdin) = child.stdin.take() else {
            terminate_child(&mut child);
            let _ = child.wait();
            return Err(MemoryError::invalid(
                "reconstruction command stdin was unavailable",
            ));
        };

        let deadline = Instant::now() + Duration::from_millis(self.config.timeout_ms);
        let writer = thread::spawn(move || -> MemoryResult<()> {
            match stdin.write_all(&input) {
                Ok(()) => {}
                Err(err) if err.kind() == ErrorKind::BrokenPipe => {}
                Err(err) => return Err(err.into()),
            }
            drop(stdin);
            Ok(())
        });
        let mut status = None;
        loop {
            if status.is_none() {
                match child.try_wait() {
                    Ok(Some(child_status)) => {
                        status = Some(child_status);
                        terminate_child(&mut child);
                        let _ = child.wait();
                    }
                    Ok(None) => {}
                    Err(err) => {
                        terminate_child(&mut child);
                        let _ = child.wait();
                        return Err(err.into());
                    }
                }
            }
            if status.is_some() && writer.is_finished() {
                break;
            }
            if Instant::now() >= deadline {
                terminate_child(&mut child);
                let _ = child.wait();
                return Err(MemoryError::invalid(format!(
                    "reconstruction command exceeded timeout_ms {}",
                    self.config.timeout_ms
                )));
            }
            thread::sleep(Duration::from_millis(5));
        }
        let status = status.expect("command status should be set before reader threads finish");
        join_writer_thread(writer)?;
        let stdout = output_files.read_stdout()?;
        let stderr = output_files.read_stderr()?;
        if !status.success() {
            return Err(MemoryError::invalid(format!(
                "reconstruction command exited with status {status}: {}",
                String::from_utf8_lossy(&stderr)
            )));
        }
        String::from_utf8(stdout).map_err(|err| {
            MemoryError::invalid(format!("reconstruction output is not UTF-8: {err}"))
        })
    }
}

fn join_writer_thread(handle: thread::JoinHandle<MemoryResult<()>>) -> MemoryResult<()> {
    handle
        .join()
        .map_err(|_| MemoryError::invalid("reconstruction command stdin writer panicked"))?
}

struct CommandOutputFiles {
    stdout: PathBuf,
    stderr: PathBuf,
}

impl CommandOutputFiles {
    fn create() -> MemoryResult<Self> {
        let base = env::temp_dir();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        for attempt in 0..100_u32 {
            let prefix = format!(
                "beater-memory-reconstructor-{}-{now}-{attempt}",
                std::process::id()
            );
            let stdout = base.join(format!("{prefix}.stdout"));
            let stderr = base.join(format!("{prefix}.stderr"));
            match (
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&stdout),
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&stderr),
            ) {
                (Ok(_), Ok(_)) => return Ok(Self { stdout, stderr }),
                (Ok(_), Err(err)) => {
                    let _ = fs::remove_file(&stdout);
                    if err.kind() != std::io::ErrorKind::AlreadyExists {
                        return Err(err.into());
                    }
                }
                (Err(err), Ok(_)) => {
                    let _ = fs::remove_file(&stderr);
                    if err.kind() != std::io::ErrorKind::AlreadyExists {
                        return Err(err.into());
                    }
                }
                (Err(left), Err(right)) => {
                    if left.kind() != std::io::ErrorKind::AlreadyExists {
                        return Err(left.into());
                    }
                    if right.kind() != std::io::ErrorKind::AlreadyExists {
                        return Err(right.into());
                    }
                }
            }
        }
        Err(MemoryError::invalid(
            "failed to allocate reconstruction command output files",
        ))
    }

    fn stdout_file(&self) -> MemoryResult<File> {
        OpenOptions::new()
            .append(true)
            .open(&self.stdout)
            .map_err(Into::into)
    }

    fn stderr_file(&self) -> MemoryResult<File> {
        OpenOptions::new()
            .append(true)
            .open(&self.stderr)
            .map_err(Into::into)
    }

    fn read_stdout(&self) -> MemoryResult<Vec<u8>> {
        fs::read(&self.stdout).map_err(Into::into)
    }

    fn read_stderr(&self) -> MemoryResult<Vec<u8>> {
        fs::read(&self.stderr).map_err(Into::into)
    }
}

impl Drop for CommandOutputFiles {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.stdout);
        let _ = fs::remove_file(&self.stderr);
    }
}

fn validate_command_path(command: &Path) -> MemoryResult<()> {
    if command.is_absolute() || command.components().count() > 1 {
        validate_command_file(command)
    } else if let Some(path) = env::var_os("PATH") {
        for dir in env::split_paths(&path) {
            let candidate = dir.join(command);
            if command_file_is_executable(&candidate) {
                return Ok(());
            }
        }
        Err(MemoryError::invalid(format!(
            "reconstruction command {:?} was not found in PATH or is not executable",
            command
        )))
    } else {
        Err(MemoryError::invalid(format!(
            "reconstruction command {:?} was not found because PATH is unset",
            command
        )))
    }
}

fn validate_command_file(command: &Path) -> MemoryResult<()> {
    if command_file_is_executable(command) {
        Ok(())
    } else {
        match fs::metadata(command) {
            Ok(metadata) if !metadata.is_file() => Err(MemoryError::invalid(format!(
                "reconstruction command {:?} is not a file",
                command
            ))),
            Ok(_) => Err(MemoryError::invalid(format!(
                "reconstruction command {:?} is not executable",
                command
            ))),
            Err(err) => Err(MemoryError::invalid(format!(
                "reconstruction command {:?} is not accessible: {err}",
                command
            ))),
        }
    }
}

fn command_file_is_executable(command: &Path) -> bool {
    let Ok(metadata) = fs::metadata(command) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderReconstructionOutput {
    decision: ProviderReconstructionDecision,
    #[serde(default)]
    node_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ProviderReconstructionDecision {
    Accept,
    Prune,
    Stop,
}

fn parse_provider_reconstruction_output(
    raw: &str,
    step: &ReconstructionStep,
) -> Result<ReconstructionDecision, String> {
    let output: ProviderReconstructionOutput =
        serde_json::from_str(raw).map_err(|err| format!("invalid provider JSON: {err}"))?;
    match output.decision {
        ProviderReconstructionDecision::Accept => {
            let node_id = required_candidate_id(output.node_id, step, "accept")?;
            Ok(ReconstructionDecision::Accept { node_id })
        }
        ProviderReconstructionDecision::Prune => {
            let node_id = required_candidate_id(output.node_id, step, "prune")?;
            Ok(ReconstructionDecision::Prune { node_id })
        }
        ProviderReconstructionDecision::Stop => {
            if output.node_id.is_some() {
                return Err("stop decision must not include node_id".to_string());
            }
            Ok(ReconstructionDecision::Stop)
        }
    }
}

fn required_candidate_id(
    node_id: Option<String>,
    step: &ReconstructionStep,
    decision: &str,
) -> Result<String, String> {
    let node_id = node_id.ok_or_else(|| format!("{decision} decision requires node_id"))?;
    if !step
        .candidates
        .iter()
        .any(|candidate| candidate.node_id == node_id)
    {
        return Err(format!(
            "{decision} decision node_id {node_id:?} is not in candidates"
        ));
    }
    Ok(node_id)
}

fn estimate_reconstruction_input_tokens(step: &ReconstructionStep) -> u32 {
    estimate_tokens(&step.question)
        + step
            .candidates
            .iter()
            .map(|candidate| estimate_tokens(&candidate.text))
            .sum::<u32>()
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct StaticProvider {
        raw: Result<String, &'static str>,
    }

    impl ReconstructionProvider for StaticProvider {
        fn decide(&self, _step: &ReconstructionStep) -> MemoryResult<String> {
            self.raw
                .clone()
                .map_err(|err| MemoryError::invalid(err.to_string()))
        }
    }

    fn step() -> ReconstructionStep {
        ReconstructionStep {
            question: "which runbook fixes checkout?".to_string(),
            step_index: 0,
            expanded_node_id: "node_seed".to_string(),
            remaining_tokens: 100,
            candidates: vec![ReconstructionCandidate {
                node_id: "node_candidate".to_string(),
                kind: MemoryNodeKind::Procedure,
                text: "Run the checkout database migration.".to_string(),
                score: 0.8,
                token_estimate: 8,
            }],
        }
    }

    #[test]
    fn provider_reconstructor_accepts_valid_candidate() -> MemoryResult<()> {
        let reconstructor = ProviderReconstructor::new(StaticProvider {
            raw: Ok(serde_json::json!({
                "decision": "accept",
                "node_id": "node_candidate"
            })
            .to_string()),
        });

        let outcome = reconstructor.decide(&step())?;

        assert_eq!(
            outcome.decision,
            ReconstructionDecision::Accept {
                node_id: "node_candidate".to_string()
            }
        );
        assert_eq!(outcome.metrics.provider_calls, 1);
        assert_eq!(outcome.metrics.provider_errors, 0);
        assert!(outcome.metrics.input_tokens > 0);
        assert!(outcome.metrics.output_tokens > 0);
        Ok(())
    }

    #[test]
    fn provider_reconstructor_prunes_valid_candidate() -> MemoryResult<()> {
        let reconstructor = ProviderReconstructor::new(StaticProvider {
            raw: Ok(serde_json::json!({
                "decision": "prune",
                "node_id": "node_candidate"
            })
            .to_string()),
        });

        let outcome = reconstructor.decide(&step())?;

        assert_eq!(
            outcome.decision,
            ReconstructionDecision::Prune {
                node_id: "node_candidate".to_string()
            }
        );
        assert_eq!(outcome.metrics.provider_calls, 1);
        assert_eq!(outcome.metrics.provider_errors, 0);
        assert_eq!(outcome.metrics.schema_errors, 0);
        Ok(())
    }

    #[test]
    fn provider_reconstructor_stops_on_stop_decision() -> MemoryResult<()> {
        let reconstructor = ProviderReconstructor::new(StaticProvider {
            raw: Ok(serde_json::json!({
                "decision": "stop"
            })
            .to_string()),
        });

        let outcome = reconstructor.decide(&step())?;

        assert_eq!(outcome.decision, ReconstructionDecision::Stop);
        assert_eq!(outcome.metrics.provider_calls, 1);
        assert_eq!(outcome.metrics.provider_errors, 0);
        assert_eq!(outcome.metrics.schema_errors, 0);
        Ok(())
    }

    #[test]
    fn provider_reconstructor_stops_on_invalid_candidate() -> MemoryResult<()> {
        let reconstructor = ProviderReconstructor::new(StaticProvider {
            raw: Ok(serde_json::json!({
                "decision": "accept",
                "node_id": "missing"
            })
            .to_string()),
        });

        let outcome = reconstructor.decide(&step())?;

        assert_eq!(outcome.decision, ReconstructionDecision::Stop);
        assert_eq!(outcome.metrics.provider_calls, 1);
        assert_eq!(outcome.metrics.provider_errors, 0);
        assert_eq!(outcome.metrics.schema_errors, 1);
        Ok(())
    }

    #[test]
    fn provider_reconstructor_stops_on_malformed_json() -> MemoryResult<()> {
        let reconstructor = ProviderReconstructor::new(StaticProvider {
            raw: Ok("not json".to_string()),
        });

        let outcome = reconstructor.decide(&step())?;

        assert_eq!(outcome.decision, ReconstructionDecision::Stop);
        assert_eq!(outcome.metrics.provider_calls, 1);
        assert_eq!(outcome.metrics.provider_errors, 0);
        assert_eq!(outcome.metrics.schema_errors, 1);
        Ok(())
    }

    #[test]
    fn provider_reconstructor_stops_on_transport_error() -> MemoryResult<()> {
        let reconstructor = ProviderReconstructor::new(StaticProvider {
            raw: Err("timeout"),
        });

        let outcome = reconstructor.decide(&step())?;

        assert_eq!(outcome.decision, ReconstructionDecision::Stop);
        assert_eq!(outcome.metrics.provider_calls, 1);
        assert_eq!(outcome.metrics.provider_errors, 1);
        assert_eq!(outcome.metrics.schema_errors, 0);
        Ok(())
    }

    #[test]
    fn command_reconstructor_rejects_missing_command() {
        let err = CommandReconstructionProviderConfig::new(
            "/definitely/missing/beater-memory-reconstructor",
        )
        .validate()
        .unwrap_err();

        assert!(err.to_string().contains("not accessible"));
    }

    #[cfg(unix)]
    fn shell_provider(script: &str) -> CommandReconstructionProvider {
        CommandReconstructionProvider::new(
            CommandReconstructionProviderConfig::new("/bin/sh")
                .with_args(["-c".to_string(), script.to_string()])
                .with_timeout_ms(1_000),
        )
    }

    #[cfg(unix)]
    #[test]
    fn command_reconstructor_runs_command_with_args() -> MemoryResult<()> {
        let provider = CommandReconstructionProvider::new(
            CommandReconstructionProviderConfig::new("/bin/sh")
                .with_args([
                    "-c",
                    "cat >/dev/null; printf '{\"decision\":\"%s\"}' \"$1\"",
                    "sh",
                    "stop",
                ])
                .with_timeout_ms(1_000),
        );

        let raw = provider.decide(&step())?;

        assert_eq!(raw, "{\"decision\":\"stop\"}");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn command_reconstructor_errors_on_nonzero_exit() {
        let provider = shell_provider("cat >/dev/null; echo provider failed >&2; exit 42");

        let err = provider.decide(&step()).unwrap_err();

        assert!(err.to_string().contains("exited with status"));
        assert!(err.to_string().contains("provider failed"));
    }

    #[cfg(unix)]
    #[test]
    fn command_reconstructor_times_out_running_child() {
        let provider = CommandReconstructionProvider::new(
            CommandReconstructionProviderConfig::new("/bin/sh")
                .with_args(["-c", "sleep 2"])
                .with_timeout_ms(50),
        );

        let err = provider.decide(&step()).unwrap_err();

        assert!(err.to_string().contains("exceeded timeout_ms"));
    }

    #[cfg(unix)]
    #[test]
    fn command_reconstructor_does_not_block_on_lingering_grandchild_output() -> MemoryResult<()> {
        let provider = CommandReconstructionProvider::new(
            CommandReconstructionProviderConfig::new("/bin/sh")
                .with_args(["-c", "sleep 2 &"])
                .with_timeout_ms(100),
        );
        let started = Instant::now();

        let raw = provider.decide(&step())?;

        assert!(started.elapsed() < Duration::from_millis(750));
        assert_eq!(raw, "");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn command_reconstructor_rejects_invalid_utf8() {
        let provider = shell_provider("cat >/dev/null; printf '\\377'");

        let err = provider.decide(&step()).unwrap_err();

        assert!(err.to_string().contains("UTF-8"));
    }
}
