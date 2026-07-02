use std::{
    collections::HashMap,
    future::Future,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Query, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::time::{Duration, timeout};

use crate::{
    MemoryEngine, ProjectReport,
    error::{MemoryError, MemoryResult},
    model::{MemoryAnswer, MemoryMode, MemoryNodeKind, MemoryQuery, MemoryScope},
    store::{
        AuditEvent, AuditRecord, LedgerEvent, MaintenanceOptions, MaintenanceReport, StoreHealth,
        StoreStats,
    },
    text::{now_unix_ms, stable_id},
};

const DEFAULT_MAX_BODY_BYTES: usize = 1024 * 1024;
const DEFAULT_MAX_PROJECT_LIMIT: usize = 10_000;
const DEFAULT_MAX_QUERY_TOKENS: u32 = 8_000;
const DEFAULT_MAX_AUDIT_LIMIT: usize = 500;
const DEFAULT_MAX_REQUESTS_PER_MINUTE: u32 = 600;
const DEFAULT_MAX_CONCURRENT_DB_TASKS: usize = 32;
const DEFAULT_DB_TASK_TIMEOUT_MS: u64 = 30_000;
const RATE_LIMIT_WINDOW_MS: i64 = 60_000;

/// HTTP server configuration.
#[derive(Clone, Debug)]
pub struct MemoryServerConfig {
    pub db_path: PathBuf,
    pub bind_addr: SocketAddr,
    pub bearer_token: Option<String>,
    pub max_body_bytes: usize,
    pub max_project_limit: usize,
    pub max_query_tokens: u32,
    pub max_audit_limit: usize,
    pub max_requests_per_minute: u32,
    pub max_concurrent_db_tasks: usize,
    pub db_task_timeout_ms: u64,
}

impl MemoryServerConfig {
    #[must_use]
    pub fn new(db_path: impl Into<PathBuf>, bind_addr: SocketAddr) -> Self {
        Self {
            db_path: db_path.into(),
            bind_addr,
            bearer_token: None,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_project_limit: DEFAULT_MAX_PROJECT_LIMIT,
            max_query_tokens: DEFAULT_MAX_QUERY_TOKENS,
            max_audit_limit: DEFAULT_MAX_AUDIT_LIMIT,
            max_requests_per_minute: DEFAULT_MAX_REQUESTS_PER_MINUTE,
            max_concurrent_db_tasks: DEFAULT_MAX_CONCURRENT_DB_TASKS,
            db_task_timeout_ms: DEFAULT_DB_TASK_TIMEOUT_MS,
        }
    }

    #[must_use]
    pub fn with_bearer_token(mut self, bearer_token: impl Into<String>) -> Self {
        self.bearer_token = Some(bearer_token.into());
        self
    }

    #[must_use]
    pub fn with_limits(
        mut self,
        max_body_bytes: usize,
        max_project_limit: usize,
        max_query_tokens: u32,
    ) -> Self {
        self.max_body_bytes = max_body_bytes;
        self.max_project_limit = max_project_limit;
        self.max_query_tokens = max_query_tokens;
        self
    }

    #[must_use]
    pub fn with_audit_limit(mut self, max_audit_limit: usize) -> Self {
        self.max_audit_limit = max_audit_limit;
        self
    }

    #[must_use]
    pub fn with_rate_limit(mut self, max_requests_per_minute: u32) -> Self {
        self.max_requests_per_minute = max_requests_per_minute;
        self
    }

    #[must_use]
    pub fn with_db_concurrency_limit(mut self, max_concurrent_db_tasks: usize) -> Self {
        self.max_concurrent_db_tasks = max_concurrent_db_tasks;
        self
    }

    #[must_use]
    pub fn with_db_task_timeout_ms(mut self, db_task_timeout_ms: u64) -> Self {
        self.db_task_timeout_ms = db_task_timeout_ms;
        self
    }
}

#[derive(Clone)]
struct MemoryServerState {
    config: Arc<MemoryServerConfig>,
    metrics: Arc<Mutex<ServiceMetricsSnapshot>>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
    db_permits: Arc<Semaphore>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveResponse {
    pub status: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadyResponse {
    pub status: String,
    pub database: String,
    pub message: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RememberHttpRequest {
    pub tenant_id: String,
    pub project_id: String,
    pub environment_id: Option<String>,
    pub kind: MemoryNodeKind,
    pub text: String,
    pub idempotency_key: Option<String>,
    pub project: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RememberHttpResponse {
    pub ingested: bool,
    pub project: Option<ProjectReport>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectHttpRequest {
    pub limit: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryHttpRequest {
    pub question: String,
    pub scope: MemoryScope,
    pub max_tokens: Option<u32>,
    pub require_fresh: Option<bool>,
    pub modes: Option<Vec<MemoryMode>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaintenanceHttpRequest {
    pub vacuum: Option<bool>,
    pub repair_orphans: Option<bool>,
    pub prune_audit_before_unix_ms: Option<i64>,
    pub retain_latest_audit_events: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditHttpQuery {
    pub limit: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditHttpResponse {
    pub events: Vec<AuditEvent>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceMetricsSnapshot {
    pub started_at_unix_ms: i64,
    pub total_requests: u64,
    pub authorized_requests: u64,
    pub unauthorized_requests: u64,
    pub rate_limited_requests: u64,
    pub db_busy_requests: u64,
    pub db_timeout_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub live_requests: u64,
    pub ready_requests: u64,
    pub health_requests: u64,
    pub stats_requests: u64,
    pub remember_requests: u64,
    pub project_requests: u64,
    pub query_requests: u64,
    pub maintenance_requests: u64,
    pub metrics_requests: u64,
    pub audit_requests: u64,
}

impl ServiceMetricsSnapshot {
    fn new() -> Self {
        Self {
            started_at_unix_ms: now_unix_ms(),
            total_requests: 0,
            authorized_requests: 0,
            unauthorized_requests: 0,
            rate_limited_requests: 0,
            db_busy_requests: 0,
            db_timeout_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            live_requests: 0,
            ready_requests: 0,
            health_requests: 0,
            stats_requests: 0,
            remember_requests: 0,
            project_requests: 0,
            query_requests: 0,
            maintenance_requests: 0,
            metrics_requests: 0,
            audit_requests: 0,
        }
    }

    fn record_started(&mut self, action: &str) {
        self.total_requests += 1;
        match action {
            "livez" => self.live_requests += 1,
            "readyz" => self.ready_requests += 1,
            "health" => self.health_requests += 1,
            "stats" => self.stats_requests += 1,
            "remember" => self.remember_requests += 1,
            "project" => self.project_requests += 1,
            "query" => self.query_requests += 1,
            "maintenance" => self.maintenance_requests += 1,
            "metrics" => self.metrics_requests += 1,
            "audit" => self.audit_requests += 1,
            _ => {}
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ErrorDetail {
    code: String,
    message: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
    retry_after_seconds: Option<u64>,
}

impl ApiError {
    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "unauthorized",
            message: "missing or invalid bearer token".to_string(),
            retry_after_seconds: None,
        }
    }

    fn rate_limited(retry_after_seconds: u64) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            code: "rate_limited",
            message: format!("rate limit exceeded; retry after {retry_after_seconds} seconds"),
            retry_after_seconds: Some(retry_after_seconds),
        }
    }

    fn service_busy() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "service_busy",
            message: "database work queue is full; retry later".to_string(),
            retry_after_seconds: Some(1),
        }
    }

    fn service_timeout(timeout_ms: u64) -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            code: "service_timeout",
            message: format!("database task exceeded {timeout_ms}ms"),
            retry_after_seconds: Some(1),
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "bad_request",
            message: message.into(),
            retry_after_seconds: None,
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal_error",
            message: message.into(),
            retry_after_seconds: None,
        }
    }
}

impl From<MemoryError> for ApiError {
    fn from(value: MemoryError) -> Self {
        match value {
            MemoryError::InvalidInput(message) => Self::bad_request(message),
            other => Self::internal(other.to_string()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorBody {
            error: ErrorDetail {
                code: self.code.to_string(),
                message: self.message,
            },
        };
        let mut response = (self.status, Json(body)).into_response();
        if self.status == StatusCode::UNAUTHORIZED {
            response.headers_mut().insert(
                header::WWW_AUTHENTICATE,
                header::HeaderValue::from_static("Bearer"),
            );
        }
        if let Some(retry_after_seconds) = self.retry_after_seconds {
            response.headers_mut().insert(
                header::RETRY_AFTER,
                header::HeaderValue::from_str(&retry_after_seconds.to_string())
                    .unwrap_or_else(|_| header::HeaderValue::from_static("60")),
            );
        }
        response
    }
}

#[derive(Clone, Debug)]
struct RequestContext {
    actor: String,
    action: &'static str,
    route: &'static str,
}

#[derive(Clone, Debug)]
struct RateDecision {
    allowed: bool,
    retry_after_seconds: u64,
}

#[derive(Clone, Debug)]
struct RateWindow {
    window_started_at_unix_ms: i64,
    count: u32,
}

#[derive(Clone, Debug)]
struct RateLimiter {
    max_requests: u32,
    window_ms: i64,
    windows: HashMap<String, RateWindow>,
}

impl RateLimiter {
    fn new(max_requests: u32, window_ms: i64) -> Self {
        Self {
            max_requests,
            window_ms,
            windows: HashMap::new(),
        }
    }

    fn check(&mut self, actor: &str, now_unix_ms: i64) -> RateDecision {
        if self.max_requests == 0 {
            return RateDecision {
                allowed: true,
                retry_after_seconds: 0,
            };
        }
        if self.windows.len() > 4096 {
            let oldest_live_window = now_unix_ms - (self.window_ms * 2);
            self.windows
                .retain(|_, window| window.window_started_at_unix_ms >= oldest_live_window);
        }
        let window = self
            .windows
            .entry(actor.to_string())
            .or_insert_with(|| RateWindow {
                window_started_at_unix_ms: now_unix_ms,
                count: 0,
            });
        if now_unix_ms - window.window_started_at_unix_ms >= self.window_ms {
            window.window_started_at_unix_ms = now_unix_ms;
            window.count = 0;
        }
        if window.count >= self.max_requests {
            let elapsed_ms = now_unix_ms - window.window_started_at_unix_ms;
            let remaining_ms = (self.window_ms - elapsed_ms).max(1);
            return RateDecision {
                allowed: false,
                retry_after_seconds: ((remaining_ms + 999) / 1000) as u64,
            };
        }
        window.count += 1;
        RateDecision {
            allowed: true,
            retry_after_seconds: 0,
        }
    }
}

/// Build the HTTP API router.
pub fn memory_router(config: MemoryServerConfig) -> Router {
    let max_body_bytes = config.max_body_bytes;
    let state = MemoryServerState {
        metrics: Arc::new(Mutex::new(ServiceMetricsSnapshot::new())),
        rate_limiter: Arc::new(Mutex::new(RateLimiter::new(
            config.max_requests_per_minute,
            RATE_LIMIT_WINDOW_MS,
        ))),
        db_permits: Arc::new(Semaphore::new(config.max_concurrent_db_tasks)),
        config: Arc::new(config),
    };

    Router::new()
        .route("/livez", get(livez))
        .route("/readyz", get(readyz))
        .route("/v1/health", get(health))
        .route("/v1/stats", get(stats))
        .route("/v1/metrics", get(metrics))
        .route("/v1/audit", get(audit))
        .route("/v1/remember", post(remember))
        .route("/v1/project", post(project))
        .route("/v1/query", post(query))
        .route("/v1/maintenance", post(maintenance))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
}

/// Run the HTTP server until Ctrl-C or SIGTERM on Unix.
pub async fn serve(config: MemoryServerConfig) -> MemoryResult<()> {
    serve_with_shutdown(config, shutdown_signal()).await
}

/// Run the HTTP server until the provided shutdown future resolves.
pub async fn serve_with_shutdown<F>(config: MemoryServerConfig, shutdown: F) -> MemoryResult<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let bind_addr = config.bind_addr;
    let router = memory_router(config);
    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(err) = tokio::signal::ctrl_c().await {
            eprintln!("failed to install Ctrl-C shutdown handler: {err}");
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    {
        let terminate = async {
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(mut signal) => {
                    signal.recv().await;
                }
                Err(err) => {
                    eprintln!("failed to install SIGTERM shutdown handler: {err}");
                    std::future::pending::<()>().await;
                }
            }
        };
        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }
    }

    #[cfg(not(unix))]
    {
        ctrl_c.await;
    }
}

async fn livez(State(state): State<MemoryServerState>) -> Json<LiveResponse> {
    record_request_started(&state, "livez");
    record_request_success(&state);
    Json(LiveResponse {
        status: "ok".to_string(),
    })
}

async fn readyz(State(state): State<MemoryServerState>) -> Response {
    record_request_started(&state, "readyz");
    match run_db_task(state.clone(), |engine| engine.store().health()).await {
        Ok(health) if health_is_ready(&health) => {
            record_request_success(&state);
            (
                StatusCode::OK,
                Json(ReadyResponse {
                    status: "ok".to_string(),
                    database: "ok".to_string(),
                    message: None,
                }),
            )
                .into_response()
        }
        Ok(_) => {
            record_request_failure(&state);
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ReadyResponse {
                    status: "not_ready".to_string(),
                    database: "unhealthy".to_string(),
                    message: Some("database health check failed".to_string()),
                }),
            )
                .into_response()
        }
        Err(err) => {
            record_request_failure(&state);
            ready_error_response(err)
        }
    }
}

async fn health(
    State(state): State<MemoryServerState>,
    headers: HeaderMap,
) -> Result<Json<StoreHealth>, ApiError> {
    let ctx = begin_request(&state, &headers, "health", "/v1/health").await?;
    let result = with_engine(state.clone(), |engine| engine.store().health()).await;
    finish_request(state, ctx, result, serde_json::json!({})).await
}

async fn stats(
    State(state): State<MemoryServerState>,
    headers: HeaderMap,
) -> Result<Json<StoreStats>, ApiError> {
    let ctx = begin_request(&state, &headers, "stats", "/v1/stats").await?;
    let result = with_engine(state.clone(), |engine| engine.store().stats()).await;
    finish_request(state, ctx, result, serde_json::json!({})).await
}

async fn metrics(
    State(state): State<MemoryServerState>,
    headers: HeaderMap,
) -> Result<Json<ServiceMetricsSnapshot>, ApiError> {
    let ctx = begin_request(&state, &headers, "metrics", "/v1/metrics").await?;
    finish_success(&state, &ctx, StatusCode::OK, serde_json::json!({})).await;
    Ok(Json(metrics_snapshot(&state)))
}

async fn audit(
    State(state): State<MemoryServerState>,
    headers: HeaderMap,
    Query(request): Query<AuditHttpQuery>,
) -> Result<Json<AuditHttpResponse>, ApiError> {
    let ctx = begin_request(&state, &headers, "audit", "/v1/audit").await?;
    let limit = request.limit.unwrap_or(100);
    if limit > state.config.max_audit_limit {
        return fail_request(
            &state,
            &ctx,
            ApiError::bad_request(format!(
                "audit limit {limit} exceeds configured max {}",
                state.config.max_audit_limit
            )),
        )
        .await;
    }
    let result = with_engine(state.clone(), move |engine| {
        engine
            .store()
            .recent_audit_events(limit)
            .map(|events| AuditHttpResponse { events })
    })
    .await;
    finish_request(state, ctx, result, serde_json::json!({ "limit": limit })).await
}

async fn remember(
    State(state): State<MemoryServerState>,
    headers: HeaderMap,
    Json(request): Json<RememberHttpRequest>,
) -> Result<Json<RememberHttpResponse>, ApiError> {
    let ctx = begin_request(&state, &headers, "remember", "/v1/remember").await?;
    if let Err(err) = validate_nonempty("tenant_id", &request.tenant_id) {
        return fail_request(&state, &ctx, err).await;
    }
    if let Err(err) = validate_nonempty("project_id", &request.project_id) {
        return fail_request(&state, &ctx, err).await;
    }
    if let Err(err) = validate_nonempty("text", &request.text) {
        return fail_request(&state, &ctx, err).await;
    }
    if let Some(idempotency_key) = request.idempotency_key.as_deref()
        && let Err(err) = validate_nonempty("idempotency_key", idempotency_key)
    {
        return fail_request(&state, &ctx, err).await;
    }

    let tenant_id = request.tenant_id.clone();
    let project_id = request.project_id.clone();
    let idempotency_key_hash = request
        .idempotency_key
        .as_deref()
        .map(|key| stable_id("idempotency_key", &[key.trim()]));
    let result = with_engine(state.clone(), move |engine| {
        let mut event = LedgerEvent::direct_memory_write(
            &request.tenant_id,
            &request.project_id,
            request.kind,
            request.text,
        );
        event.environment_id = request.environment_id;
        if let Some(idempotency_key) = request.idempotency_key.as_deref() {
            event.apply_idempotency_key(idempotency_key);
        }
        let ingested = engine.ingest_event(&event)?;
        let project = if request.project.unwrap_or(true) {
            Some(engine.project_pending(100)?)
        } else {
            None
        };
        Ok(RememberHttpResponse { ingested, project })
    })
    .await;
    finish_request(
        state,
        ctx,
        result,
        serde_json::json!({
            "tenant_id": tenant_id,
            "project_id": project_id,
            "idempotency_key_hash": idempotency_key_hash
        }),
    )
    .await
}

async fn project(
    State(state): State<MemoryServerState>,
    headers: HeaderMap,
    Json(request): Json<ProjectHttpRequest>,
) -> Result<Json<ProjectReport>, ApiError> {
    let ctx = begin_request(&state, &headers, "project", "/v1/project").await?;
    let limit = request.limit.unwrap_or(1000);
    if limit > state.config.max_project_limit {
        return fail_request(
            &state,
            &ctx,
            ApiError::bad_request(format!(
                "project limit {limit} exceeds configured max {}",
                state.config.max_project_limit
            )),
        )
        .await;
    }
    let result = with_engine(state.clone(), move |engine| engine.project_pending(limit)).await;
    finish_request(state, ctx, result, serde_json::json!({ "limit": limit })).await
}

async fn query(
    State(state): State<MemoryServerState>,
    headers: HeaderMap,
    Json(request): Json<QueryHttpRequest>,
) -> Result<Json<MemoryAnswer>, ApiError> {
    let ctx = begin_request(&state, &headers, "query", "/v1/query").await?;
    if let Err(err) = validate_nonempty("question", &request.question) {
        return fail_request(&state, &ctx, err).await;
    }
    if let Err(err) = validate_nonempty("tenant_id", &request.scope.tenant_id) {
        return fail_request(&state, &ctx, err).await;
    }
    if let Err(err) = validate_nonempty("project_id", &request.scope.project_id) {
        return fail_request(&state, &ctx, err).await;
    }

    let max_tokens = request.max_tokens.unwrap_or(1_200);
    if max_tokens > state.config.max_query_tokens {
        return fail_request(
            &state,
            &ctx,
            ApiError::bad_request(format!(
                "query max_tokens {max_tokens} exceeds configured max {}",
                state.config.max_query_tokens
            )),
        )
        .await;
    }
    let tenant_id = request.scope.tenant_id.clone();
    let project_id = request.scope.project_id.clone();
    let mut query = MemoryQuery::new(request.question, request.scope).with_max_tokens(max_tokens);
    if request.require_fresh.unwrap_or(false) {
        query = query.requiring_fresh();
    }
    if let Some(modes) = request.modes {
        query = query.with_modes(modes);
    }
    let result = with_engine(state.clone(), move |engine| engine.query(&query)).await;
    finish_request(
        state,
        ctx,
        result,
        serde_json::json!({
            "tenant_id": tenant_id,
            "project_id": project_id,
            "max_tokens": max_tokens
        }),
    )
    .await
}

async fn maintenance(
    State(state): State<MemoryServerState>,
    headers: HeaderMap,
    Json(request): Json<MaintenanceHttpRequest>,
) -> Result<Json<MaintenanceReport>, ApiError> {
    let ctx = begin_request(&state, &headers, "maintenance", "/v1/maintenance").await?;
    let options = MaintenanceOptions {
        vacuum: request.vacuum.unwrap_or(false),
        repair_orphans: request.repair_orphans.unwrap_or(false),
        prune_audit_before_unix_ms: request.prune_audit_before_unix_ms,
        retain_latest_audit_events: request.retain_latest_audit_events,
    };
    let detail_options = options.clone();
    let result = with_engine(state.clone(), move |engine| {
        engine.store().maintenance_with_options(options)
    })
    .await;
    finish_request(
        state,
        ctx,
        result,
        serde_json::json!({
            "vacuum": detail_options.vacuum,
            "repair_orphans": detail_options.repair_orphans,
            "prune_audit_before_unix_ms": detail_options.prune_audit_before_unix_ms,
            "retain_latest_audit_events": detail_options.retain_latest_audit_events
        }),
    )
    .await
}

async fn with_engine<T>(
    state: MemoryServerState,
    f: impl FnOnce(MemoryEngine) -> MemoryResult<T> + Send + 'static,
) -> Result<Json<T>, ApiError>
where
    T: Serialize + Send + 'static,
{
    run_db_task(state, f).await.map(Json)
}

async fn run_db_task<T>(
    state: MemoryServerState,
    f: impl FnOnce(MemoryEngine) -> MemoryResult<T> + Send + 'static,
) -> Result<T, ApiError>
where
    T: Send + 'static,
{
    let db_path = state.config.db_path.clone();
    let permit = match state.db_permits.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            record_db_busy(&state);
            return Err(ApiError::service_busy());
        }
    };
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let engine = MemoryEngine::open(db_path)?;
        f(engine)
    });
    let result = match timeout(
        Duration::from_millis(state.config.db_task_timeout_ms),
        result,
    )
    .await
    {
        Ok(join_result) => join_result.map_err(|err| ApiError::internal(err.to_string()))?,
        Err(_) => {
            record_db_timeout(&state);
            return Err(ApiError::service_timeout(state.config.db_task_timeout_ms));
        }
    };
    result.map_err(Into::into)
}

async fn begin_request(
    state: &MemoryServerState,
    headers: &HeaderMap,
    action: &'static str,
    route: &'static str,
) -> Result<RequestContext, ApiError> {
    record_request_started(state, action);
    match authorize(&state.config, headers) {
        Ok(actor) => {
            record_request_authorized(state);
            let ctx = RequestContext {
                actor,
                action,
                route,
            };
            if let Err(err) = check_rate_limit(state, &ctx.actor) {
                record_request_rejected(state, false, true);
                append_audit_event(
                    state,
                    AuditRecord {
                        actor: ctx.actor.clone(),
                        action: action.to_string(),
                        outcome: "rate_limited".to_string(),
                        route: Some(route.to_string()),
                        status_code: Some(err.status.as_u16()),
                        detail: error_detail(&err),
                    },
                )
                .await;
                return Err(err);
            }
            Ok(ctx)
        }
        Err(err) => {
            let ctx = RequestContext {
                actor: "unauthenticated".to_string(),
                action,
                route,
            };
            if let Err(rate_limit_err) = check_rate_limit(state, &ctx.actor) {
                record_request_rejected(state, true, true);
                append_audit_event(
                    state,
                    AuditRecord {
                        actor: ctx.actor,
                        action: action.to_string(),
                        outcome: "rate_limited".to_string(),
                        route: Some(route.to_string()),
                        status_code: Some(rate_limit_err.status.as_u16()),
                        detail: error_detail(&rate_limit_err),
                    },
                )
                .await;
                return Err(rate_limit_err);
            }
            record_request_rejected(state, true, false);
            append_audit_event(
                state,
                AuditRecord {
                    actor: ctx.actor,
                    action: action.to_string(),
                    outcome: "denied".to_string(),
                    route: Some(route.to_string()),
                    status_code: Some(err.status.as_u16()),
                    detail: error_detail(&err),
                },
            )
            .await;
            Err(err)
        }
    }
}

async fn finish_request<T>(
    state: MemoryServerState,
    ctx: RequestContext,
    result: Result<Json<T>, ApiError>,
    success_detail: serde_json::Value,
) -> Result<Json<T>, ApiError> {
    match result {
        Ok(response) => {
            finish_success(&state, &ctx, StatusCode::OK, success_detail).await;
            Ok(response)
        }
        Err(err) => fail_request(&state, &ctx, err).await,
    }
}

async fn fail_request<T>(
    state: &MemoryServerState,
    ctx: &RequestContext,
    err: ApiError,
) -> Result<T, ApiError> {
    finish_failure(state, ctx, &err).await;
    Err(err)
}

async fn finish_success(
    state: &MemoryServerState,
    ctx: &RequestContext,
    status: StatusCode,
    detail: serde_json::Value,
) {
    record_request_success(state);
    append_audit_event(
        state,
        AuditRecord {
            actor: ctx.actor.clone(),
            action: ctx.action.to_string(),
            outcome: "success".to_string(),
            route: Some(ctx.route.to_string()),
            status_code: Some(status.as_u16()),
            detail,
        },
    )
    .await;
}

async fn finish_failure(state: &MemoryServerState, ctx: &RequestContext, err: &ApiError) {
    record_request_failure(state);
    append_audit_event(
        state,
        AuditRecord {
            actor: ctx.actor.clone(),
            action: ctx.action.to_string(),
            outcome: "failure".to_string(),
            route: Some(ctx.route.to_string()),
            status_code: Some(err.status.as_u16()),
            detail: error_detail(err),
        },
    )
    .await;
}

async fn append_audit_event(state: &MemoryServerState, record: AuditRecord) {
    let db_path = state.config.db_path.clone();
    let permit = match state.db_permits.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            eprintln!("skipping audit event because database work queue is full");
            return;
        }
    };
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let engine = MemoryEngine::open(db_path)?;
        engine.store().append_audit(&record)?;
        Ok::<_, MemoryError>(())
    })
    .await;
    match result {
        Ok(Ok(())) => {}
        Ok(Err(err)) => eprintln!("failed to write audit event: {err}"),
        Err(err) => eprintln!("failed to join audit task: {err}"),
    }
}

fn check_rate_limit(state: &MemoryServerState, actor: &str) -> Result<(), ApiError> {
    let mut limiter = state
        .rate_limiter
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let decision = limiter.check(actor, now_unix_ms());
    if decision.allowed {
        Ok(())
    } else {
        Err(ApiError::rate_limited(decision.retry_after_seconds))
    }
}

fn metrics_snapshot(state: &MemoryServerState) -> ServiceMetricsSnapshot {
    state
        .metrics
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .clone()
}

fn record_request_started(state: &MemoryServerState, action: &str) {
    state
        .metrics
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .record_started(action);
}

fn record_request_authorized(state: &MemoryServerState) {
    state
        .metrics
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .authorized_requests += 1;
}

fn record_request_success(state: &MemoryServerState) {
    state
        .metrics
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .successful_requests += 1;
}

fn record_request_failure(state: &MemoryServerState) {
    state
        .metrics
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .failed_requests += 1;
}

fn record_request_rejected(state: &MemoryServerState, unauthorized: bool, rate_limited: bool) {
    let mut metrics = state.metrics.lock().unwrap_or_else(|err| err.into_inner());
    metrics.failed_requests += 1;
    if unauthorized {
        metrics.unauthorized_requests += 1;
    }
    if rate_limited {
        metrics.rate_limited_requests += 1;
    }
}

fn record_db_busy(state: &MemoryServerState) {
    state
        .metrics
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .db_busy_requests += 1;
}

fn record_db_timeout(state: &MemoryServerState) {
    state
        .metrics
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .db_timeout_requests += 1;
}

fn health_is_ready(health: &StoreHealth) -> bool {
    health.application_id == health.expected_application_id
        && health.schema_version == health.expected_schema_version
        && health.integrity_ok
        && health.foreign_key_violations == 0
        && health.graph_integrity_ok
}

fn ready_error_response(err: ApiError) -> Response {
    let (database, message) = match err.code {
        "service_busy" => ("busy", "database work queue is full"),
        "service_timeout" => ("timeout", "database readiness check timed out"),
        _ => ("unavailable", "database readiness check failed"),
    };
    let mut response = (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ReadyResponse {
            status: "not_ready".to_string(),
            database: database.to_string(),
            message: Some(message.to_string()),
        }),
    )
        .into_response();
    if let Some(retry_after_seconds) = err.retry_after_seconds {
        response.headers_mut().insert(
            header::RETRY_AFTER,
            header::HeaderValue::from_str(&retry_after_seconds.to_string())
                .unwrap_or_else(|_| header::HeaderValue::from_static("60")),
        );
    }
    response
}

fn error_detail(err: &ApiError) -> serde_json::Value {
    serde_json::json!({
        "code": err.code,
        "message": err.message
    })
}

fn authorize(config: &MemoryServerConfig, headers: &HeaderMap) -> Result<String, ApiError> {
    let Some(expected) = config.bearer_token.as_deref() else {
        return Ok("anonymous".to_string());
    };
    let Some(value) = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    else {
        return Err(ApiError::unauthorized());
    };
    let Some(actual) = value.strip_prefix("Bearer ") else {
        return Err(ApiError::unauthorized());
    };
    if constant_time_eq(actual.as_bytes(), expected.as_bytes()) {
        Ok(stable_id("bearer", &[actual]))
    } else {
        Err(ApiError::unauthorized())
    }
}

fn validate_nonempty(field: &str, value: &str) -> Result<(), ApiError> {
    if value.trim().is_empty() {
        Err(ApiError::bad_request(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut diff = 0_u8;
    for (left, right) in left.iter().zip(right.iter()) {
        diff |= left ^ right;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode, header},
    };
    use tower::ServiceExt;

    use super::*;

    #[tokio::test]
    async fn v1_routes_require_bearer_auth() {
        let app = memory_router(test_config().with_bearer_token("secret"));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/stats")
                    .body(Body::empty())
                    .unwrap_or_else(|err| panic!("{err}")),
            )
            .await
            .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn readyz_checks_database_without_auth() {
        let app = memory_router(test_config().with_bearer_token("secret"));

        let response = app
            .oneshot(unauthenticated_get_request("/readyz"))
            .await
            .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let ready: ReadyResponse =
            serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(ready.status, "ok");
        assert_eq!(ready.database, "ok");
    }

    #[tokio::test]
    async fn serve_with_shutdown_exits_when_shutdown_future_resolves() {
        serve_with_shutdown(test_config().with_bearer_token("secret"), async {})
            .await
            .unwrap_or_else(|err| panic!("{err}"));
    }

    #[tokio::test]
    async fn remember_and_query_work_over_http() {
        let app = memory_router(test_config().with_bearer_token("secret"));
        let remember = serde_json::json!({
            "tenant_id": "tenant",
            "project_id": "project",
            "kind": "gotcha",
            "text": "Checkout fails when DATABASE_URL is missing. Fix by setting DATABASE_URL."
        });

        let response = app
            .clone()
            .oneshot(json_request("/v1/remember", remember))
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(response.status(), StatusCode::OK);

        let query = serde_json::json!({
            "question": "How do we fix checkout database failures?",
            "scope": {"tenant_id": "tenant", "project_id": "project", "environment_id": null, "as_of_unix_ms": null},
            "max_tokens": 1200
        });
        let response = app
            .oneshot(json_request("/v1/query", query))
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let answer: MemoryAnswer =
            serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert!(answer.answer.contains("DATABASE_URL"));
        assert!(!answer.cited_spans.is_empty());
    }

    #[tokio::test]
    async fn remember_idempotency_key_deduplicates_retries() {
        let app = memory_router(test_config().with_bearer_token("secret"));
        let remember = serde_json::json!({
            "tenant_id": "tenant",
            "project_id": "project",
            "environment_id": "prod",
            "kind": "fact",
            "text": "Checkout uses DATABASE_URL.",
            "idempotency_key": "retry-1",
            "project": false
        });

        let response = app
            .clone()
            .oneshot(json_request("/v1/remember", remember.clone()))
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let first: RememberHttpResponse =
            serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert!(first.ingested);

        let response = app
            .clone()
            .oneshot(json_request("/v1/remember", remember))
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let second: RememberHttpResponse =
            serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert!(!second.ingested);

        let response = app
            .oneshot(get_request("/v1/stats"))
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let stats: StoreStats = serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(stats.ledger_events, 1);
        assert_eq!(stats.pending_events, 1);
    }

    #[tokio::test]
    async fn remember_rejects_empty_idempotency_key() {
        let app = memory_router(test_config().with_bearer_token("secret"));
        let remember = serde_json::json!({
            "tenant_id": "tenant",
            "project_id": "project",
            "kind": "fact",
            "text": "Checkout uses DATABASE_URL.",
            "idempotency_key": "   "
        });

        let response = app
            .oneshot(json_request("/v1/remember", remember))
            .await
            .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn query_limit_is_enforced() {
        let app = memory_router(
            test_config()
                .with_bearer_token("secret")
                .with_limits(4096, 10, 5),
        );
        let query = serde_json::json!({
            "question": "anything",
            "scope": {"tenant_id": "tenant", "project_id": "project", "environment_id": null, "as_of_unix_ms": null},
            "max_tokens": 6
        });

        let response = app
            .oneshot(json_request("/v1/query", query))
            .await
            .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn metrics_and_audit_are_available_over_http() {
        let app = memory_router(test_config().with_bearer_token("secret"));
        let remember = serde_json::json!({
            "tenant_id": "tenant",
            "project_id": "project",
            "kind": "fact",
            "text": "The public health route is /livez."
        });

        let response = app
            .clone()
            .oneshot(json_request("/v1/remember", remember))
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(response.status(), StatusCode::OK);

        let response = app
            .clone()
            .oneshot(get_request("/v1/metrics"))
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let metrics: ServiceMetricsSnapshot =
            serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(metrics.remember_requests, 1);
        assert_eq!(metrics.metrics_requests, 1);
        assert_eq!(metrics.successful_requests, 2);

        let response = app
            .oneshot(get_request("/v1/audit?limit=10"))
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let audit: AuditHttpResponse =
            serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert!(audit.events.iter().any(|event| {
            event.action == "remember"
                && event.outcome == "success"
                && event.status_code == Some(200)
        }));
    }

    #[tokio::test]
    async fn per_actor_rate_limit_is_enforced() {
        let app = memory_router(test_config().with_bearer_token("secret").with_rate_limit(1));

        let response = app
            .clone()
            .oneshot(get_request("/v1/stats"))
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(response.status(), StatusCode::OK);

        let response = app
            .oneshot(get_request("/v1/stats"))
            .await
            .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(response.headers().contains_key(header::RETRY_AFTER));
    }

    #[tokio::test]
    async fn db_concurrency_limit_returns_service_busy() {
        let app = memory_router(
            test_config()
                .with_bearer_token("secret")
                .with_db_concurrency_limit(0),
        );

        let response = app
            .clone()
            .oneshot(get_request("/v1/stats"))
            .await
            .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(response.headers().contains_key(header::RETRY_AFTER));

        let response = app
            .oneshot(get_request("/v1/metrics"))
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let metrics: ServiceMetricsSnapshot =
            serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(metrics.db_busy_requests, 1);
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.successful_requests, 1);
    }

    #[tokio::test]
    async fn readyz_reports_db_queue_saturation() {
        let app = memory_router(
            test_config()
                .with_bearer_token("secret")
                .with_db_concurrency_limit(0),
        );

        let response = app
            .clone()
            .oneshot(unauthenticated_get_request("/readyz"))
            .await
            .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(response.headers().contains_key(header::RETRY_AFTER));
        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let ready: ReadyResponse =
            serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(ready.status, "not_ready");
        assert_eq!(ready.database, "busy");

        let response = app
            .oneshot(get_request("/v1/metrics"))
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let metrics: ServiceMetricsSnapshot =
            serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(metrics.ready_requests, 1);
        assert_eq!(metrics.db_busy_requests, 1);
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.unauthorized_requests, 0);
    }

    #[tokio::test]
    async fn db_task_timeout_returns_gateway_timeout() {
        let app = memory_router(
            test_config()
                .with_bearer_token("secret")
                .with_db_task_timeout_ms(0),
        );

        let response = app
            .clone()
            .oneshot(get_request("/v1/stats"))
            .await
            .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert!(response.headers().contains_key(header::RETRY_AFTER));
        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let error: ErrorBody = serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(error.error.code, "service_timeout");

        let response = app
            .oneshot(get_request("/v1/metrics"))
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let metrics: ServiceMetricsSnapshot =
            serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(metrics.db_timeout_requests, 1);
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.successful_requests, 1);
    }

    #[tokio::test]
    async fn readyz_reports_unhealthy_graph_integrity() {
        let config = test_config().with_bearer_token("secret");
        let store =
            crate::SqliteMemoryStore::open(&config.db_path).unwrap_or_else(|err| panic!("{err}"));
        insert_orphan_projection_rows(&store);
        drop(store);
        let app = memory_router(config);

        let response = app
            .oneshot(unauthenticated_get_request("/readyz"))
            .await
            .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let ready: ReadyResponse =
            serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(ready.status, "not_ready");
        assert_eq!(ready.database, "unhealthy");
    }

    #[tokio::test]
    async fn maintenance_repairs_graph_orphans_over_http() {
        let config = test_config().with_bearer_token("secret");
        let store =
            crate::SqliteMemoryStore::open(&config.db_path).unwrap_or_else(|err| panic!("{err}"));
        insert_orphan_projection_rows(&store);
        assert!(
            !store
                .health()
                .unwrap_or_else(|err| panic!("{err}"))
                .graph_integrity_ok
        );
        drop(store);
        let app = memory_router(config);
        let request = serde_json::json!({
            "vacuum": false,
            "repair_orphans": true
        });

        let response = app
            .oneshot(json_request("/v1/maintenance", request))
            .await
            .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let report: MaintenanceReport =
            serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert!(report.repaired_orphans);
        assert!(!report.graph_integrity_before.is_clean());
        assert!(report.graph_integrity_after.is_clean());
        assert_eq!(report.graph_repair.memory_edges_removed, 1);
        assert_eq!(report.graph_repair.node_spans_removed, 1);
        assert_eq!(report.graph_repair.cue_index_entries_removed, 1);
    }

    #[tokio::test]
    async fn maintenance_prunes_audit_events_over_http() {
        let config = test_config().with_bearer_token("secret");
        let db_path = config.db_path.clone();
        let store =
            crate::SqliteMemoryStore::open(&config.db_path).unwrap_or_else(|err| panic!("{err}"));
        insert_audit_event(&store, 1_000, "oldest");
        insert_audit_event(&store, 2_000, "old");
        insert_audit_event(&store, 3_000, "new");
        insert_audit_event(&store, 4_000, "newest");
        drop(store);
        let app = memory_router(config);
        let request = serde_json::json!({
            "prune_audit_before_unix_ms": 2_500,
            "retain_latest_audit_events": 1
        });

        let response = app
            .oneshot(json_request("/v1/maintenance", request))
            .await
            .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 128 * 1024)
            .await
            .unwrap_or_else(|err| panic!("{err}"));
        let report: MaintenanceReport =
            serde_json::from_slice(&body).unwrap_or_else(|err| panic!("{err}"));
        assert!(report.pruned_audit_events);
        assert_eq!(report.audit_prune.audit_events_removed, 3);

        let store = crate::SqliteMemoryStore::open(&db_path).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(
            store
                .stats()
                .unwrap_or_else(|err| panic!("{err}"))
                .audit_events,
            2
        );
        let events = store
            .recent_audit_events(10)
            .unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(events[0].action, "maintenance");
        assert_eq!(events[1].action, "newest");
    }

    fn json_request(path: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(path)
            .header(header::AUTHORIZATION, "Bearer secret")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn get_request(path: &str) -> Request<Body> {
        Request::builder()
            .method("GET")
            .uri(path)
            .header(header::AUTHORIZATION, "Bearer secret")
            .body(Body::empty())
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn unauthenticated_get_request(path: &str) -> Request<Body> {
        Request::builder()
            .method("GET")
            .uri(path)
            .body(Body::empty())
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn test_config() -> MemoryServerConfig {
        let db_path = std::env::temp_dir().join(format!(
            "beater-memory-api-test-{}.db",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        MemoryServerConfig::new(
            db_path,
            "127.0.0.1:0".parse().unwrap_or_else(|err| panic!("{err}")),
        )
    }

    fn insert_audit_event(
        store: &crate::SqliteMemoryStore,
        occurred_at_unix_ms: i64,
        action: &str,
    ) {
        store
            .connection()
            .execute(
                "
                INSERT INTO audit_events(
                    occurred_at_unix_ms, actor, action, outcome, route, status_code, detail_json
                ) VALUES (?1, 'test-actor', ?2, 'success', '/test', 200, '{}')
                ",
                rusqlite::params![occurred_at_unix_ms, action],
            )
            .unwrap_or_else(|err| panic!("{err}"));
    }

    fn insert_orphan_projection_rows(store: &crate::SqliteMemoryStore) {
        store
            .connection()
            .execute_batch(
                "
                INSERT INTO memory_edges(
                    tenant_id, project_id, from_node_id, to_node_id, kind, weight, created_at_unix_ms
                ) VALUES (
                    'tenant', 'project', 'missing-from', 'missing-to', 'mentions', 1.0, 1
                );
                INSERT INTO node_spans(node_id, tenant_id, project_id, trace_id, span_id, seq)
                VALUES ('missing-node', 'tenant', 'project', 'trace', 'span', 1);
                INSERT INTO cue_index(term, node_id, tenant_id, project_id, weight)
                VALUES ('missing', 'missing-node', 'tenant', 'project', 1.0);
                ",
            )
            .unwrap_or_else(|err| panic!("{err}"));
    }
}
