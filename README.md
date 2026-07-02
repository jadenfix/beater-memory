# beater-memory

Agent-first memory for Beater.

This repo is the standalone workspace for the memory engine described in
[`research/agent-memory.md`](research/agent-memory.md). The core thesis is that
agent memory should be a projection over an append-only trace ledger, not a
plain vector database.

## Shape

```text
ledgered agent traces
  -> offline distiller
  -> typed temporal graph and substores
  -> tiered read path
  -> compact answer with provenance, contradictions, and token budget
```

The initial implementation starts small:

- no embeddings in the MVP; Tantivy-style lexical cue seeding comes first
- graph traversal and activation ranking are LLM-free by default
- active reconstruction is an escalation tier, not the normal path
- every returned memory must cite its source trace/span evidence
- contradicted facts are invalidated bitemporally, never deleted

## Crate

`crates/beater-memory` currently contains the dependency-free public primitives:

- `MemoryQuery`
- `MemoryAnswer`
- `MemoryTier`
- `MemoryNodeKind`
- `MemoryEdgeKind`
- `BeliefRevisionOp`
- evidence token budgeting helpers

Run checks:

```bash
cargo test
```

## Design Constraints

- Memory is `write / manage / read`, not just `retrieve`.
- The write hot path should stay append-only and robust.
- LLM distillation happens off-path and must use constrained schemas before
  writing projections.
- Retrieval returns an answer-shaped evidence bundle, not raw chunks.
- Token cost, read latency, write amplification, and context pollution are
  first-class metrics.
