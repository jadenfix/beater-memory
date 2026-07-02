use std::{net::SocketAddr, path::PathBuf, sync::Arc};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use crate::{
    MemoryEngine, ProjectReport,
    error::{MemoryError, MemoryResult},
    model::{MemoryAnswer, MemoryMode, MemoryNodeKind, MemoryQuery, MemoryScope},
    store::{LedgerEvent, MaintenanceReport, StoreHealth, StoreStats},
};

const DEFAULT_MAX_BODY_BYTES: usize = 1024 * 1024;
const DEFAULT_MAX_PROJECT_LIMIT: usize = 10_000;
const DEFAULT_MAX_QUERY_TOKENS: u32 = 8_000;

/// HTTP server configuration.
#[derive(Clone, Debug)]
pub struct MemoryServerConfig {
    pub db_path: PathBuf,
    pub bind_addr: SocketAddr,
    pub bearer_token: Option<String>,
    pub max_body_bytes: usize,
    pub max_project_limit: usize,
    pub max_query_tokens: u32,
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
}

#[derive(Clone)]
struct MemoryServerState {
    config: Arc<MemoryServerConfig>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveResponse {
    pub status: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RememberHttpRequest {
    pub tenant_id: String,
    pub project_id: String,
    pub environment_id: Option<String>,
    pub kind: MemoryNodeKind,
    pub text: String,
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
}

impl ApiError {
    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "unauthorized",
            message: "missing or invalid bearer token".to_string(),
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "bad_request",
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal_error",
            message: message.into(),
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
        response
    }
}

/// Build the HTTP API router.
pub fn memory_router(config: MemoryServerConfig) -> Router {
    let max_body_bytes = config.max_body_bytes;
    let state = MemoryServerState {
        config: Arc::new(config),
    };

    Router::new()
        .route("/livez", get(livez))
        .route("/v1/health", get(health))
        .route("/v1/stats", get(stats))
        .route("/v1/remember", post(remember))
        .route("/v1/project", post(project))
        .route("/v1/query", post(query))
        .route("/v1/maintenance", post(maintenance))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
}

/// Run the HTTP server until Ctrl-C.
pub async fn serve(config: MemoryServerConfig) -> MemoryResult<()> {
    let bind_addr = config.bind_addr;
    let router = memory_router(config);
    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

async fn livez() -> Json<LiveResponse> {
    Json(LiveResponse {
        status: "ok".to_string(),
    })
}

async fn health(
    State(state): State<MemoryServerState>,
    headers: HeaderMap,
) -> Result<Json<StoreHealth>, ApiError> {
    authorize(&state.config, &headers)?;
    with_engine(state, |engine| engine.store().health()).await
}

async fn stats(
    State(state): State<MemoryServerState>,
    headers: HeaderMap,
) -> Result<Json<StoreStats>, ApiError> {
    authorize(&state.config, &headers)?;
    with_engine(state, |engine| engine.store().stats()).await
}

async fn remember(
    State(state): State<MemoryServerState>,
    headers: HeaderMap,
    Json(request): Json<RememberHttpRequest>,
) -> Result<Json<RememberHttpResponse>, ApiError> {
    authorize(&state.config, &headers)?;
    validate_nonempty("tenant_id", &request.tenant_id)?;
    validate_nonempty("project_id", &request.project_id)?;
    validate_nonempty("text", &request.text)?;

    with_engine(state, move |engine| {
        let mut event = LedgerEvent::direct_memory_write(
            &request.tenant_id,
            &request.project_id,
            request.kind,
            request.text,
        );
        event.environment_id = request.environment_id;
        let ingested = engine.ingest_event(&event)?;
        let project = if request.project.unwrap_or(true) {
            Some(engine.project_pending(100)?)
        } else {
            None
        };
        Ok(RememberHttpResponse { ingested, project })
    })
    .await
}

async fn project(
    State(state): State<MemoryServerState>,
    headers: HeaderMap,
    Json(request): Json<ProjectHttpRequest>,
) -> Result<Json<ProjectReport>, ApiError> {
    authorize(&state.config, &headers)?;
    let limit = request.limit.unwrap_or(1000);
    if limit > state.config.max_project_limit {
        return Err(ApiError::bad_request(format!(
            "project limit {limit} exceeds configured max {}",
            state.config.max_project_limit
        )));
    }
    with_engine(state, move |engine| engine.project_pending(limit)).await
}

async fn query(
    State(state): State<MemoryServerState>,
    headers: HeaderMap,
    Json(request): Json<QueryHttpRequest>,
) -> Result<Json<MemoryAnswer>, ApiError> {
    authorize(&state.config, &headers)?;
    validate_nonempty("question", &request.question)?;
    validate_nonempty("tenant_id", &request.scope.tenant_id)?;
    validate_nonempty("project_id", &request.scope.project_id)?;

    let max_tokens = request.max_tokens.unwrap_or(1_200);
    if max_tokens > state.config.max_query_tokens {
        return Err(ApiError::bad_request(format!(
            "query max_tokens {max_tokens} exceeds configured max {}",
            state.config.max_query_tokens
        )));
    }
    let mut query = MemoryQuery::new(request.question, request.scope).with_max_tokens(max_tokens);
    if request.require_fresh.unwrap_or(false) {
        query = query.requiring_fresh();
    }
    if let Some(modes) = request.modes {
        query = query.with_modes(modes);
    }
    with_engine(state, move |engine| engine.query(&query)).await
}

async fn maintenance(
    State(state): State<MemoryServerState>,
    headers: HeaderMap,
    Json(request): Json<MaintenanceHttpRequest>,
) -> Result<Json<MaintenanceReport>, ApiError> {
    authorize(&state.config, &headers)?;
    with_engine(state, move |engine| {
        engine.store().maintenance(request.vacuum.unwrap_or(false))
    })
    .await
}

async fn with_engine<T>(
    state: MemoryServerState,
    f: impl FnOnce(MemoryEngine) -> MemoryResult<T> + Send + 'static,
) -> Result<Json<T>, ApiError>
where
    T: Serialize + Send + 'static,
{
    let db_path = state.config.db_path.clone();
    let result = tokio::task::spawn_blocking(move || {
        let engine = MemoryEngine::open(db_path)?;
        f(engine)
    })
    .await
    .map_err(|err| ApiError::internal(err.to_string()))?;
    result.map(Json).map_err(Into::into)
}

fn authorize(config: &MemoryServerConfig, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(expected) = config.bearer_token.as_deref() else {
        return Ok(());
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
        Ok(())
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

    fn json_request(path: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(path)
            .header(header::AUTHORIZATION, "Bearer secret")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
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
}
