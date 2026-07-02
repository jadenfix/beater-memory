use std::{fmt, io};

/// Result type used by the memory engine.
pub type MemoryResult<T> = Result<T, MemoryError>;

/// Typed error boundary for storage, parsing, and invalid caller input.
#[derive(Debug)]
pub enum MemoryError {
    Io(io::Error),
    Sqlite(rusqlite::Error),
    Json(serde_json::Error),
    InvalidInput(String),
}

impl MemoryError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Sqlite(err) => write!(f, "sqlite error: {err}"),
            Self::Json(err) => write!(f, "json error: {err}"),
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
        }
    }
}

impl std::error::Error for MemoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Sqlite(err) => Some(err),
            Self::Json(err) => Some(err),
            Self::InvalidInput(_) => None,
        }
    }
}

impl From<io::Error> for MemoryError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<rusqlite::Error> for MemoryError {
    fn from(value: rusqlite::Error) -> Self {
        Self::Sqlite(value)
    }
}

impl From<serde_json::Error> for MemoryError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}
