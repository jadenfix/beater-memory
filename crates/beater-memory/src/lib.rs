//! Agent-first memory for Beater.
//!
//! The crate is an end-to-end local memory engine: append ledger observations,
//! project them into a typed temporal graph, and answer under an explicit token
//! budget with provenance and contradiction warnings.

mod distill;
mod engine;
mod error;
mod graph;
mod imports;
mod model;
mod server;
mod store;
mod text;

pub use distill::{Distiller, HeuristicDistiller};
pub use engine::{MemoryEngine, ProjectReport, ProjectionRebuildReport};
pub use error::{MemoryError, MemoryResult};
pub use imports::{
    BeaterJsImportReport, BeaterJsJournal, CanonicalJsonlImportReport, import_canonical_jsonl,
};
pub use model::{
    ActivationWeights, BeliefRevisionOp, CitedSpan, Contradiction, DistilledMemory, Evidence,
    MemoryAnswer, MemoryEdgeKind, MemoryMode, MemoryNodeKind, MemoryQuery, MemoryScope, MemoryTier,
    StaleAssumption, blend_activation, budget_evidence, estimate_tokens,
};
pub use server::{
    AuditHttpQuery, AuditHttpResponse, LiveResponse, MaintenanceHttpRequest, MemoryServerConfig,
    ProjectHttpRequest, QueryHttpRequest, ReadyResponse, RememberHttpRequest, RememberHttpResponse,
    ServiceMetricsSnapshot, memory_router, serve,
};
pub use store::{
    AuditEvent, AuditPruneReport, AuditRecord, BackupReport, GraphIntegrityReport,
    GraphRepairReport, LedgerEvent, MaintenanceOptions, MaintenanceReport, MemoryEdge, MemoryNode,
    ProjectionResetReport, RestoreReport, SqliteMemoryStore, StoreHealth, StoreStats,
};
