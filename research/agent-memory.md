# Agent Memory From First Principles — Research Report & Engine Design

**Date:** 2026-07-02
**Method:** Deep-research workflow — 5 parallel search angles, 23 primary sources fetched (mostly arXiv), 113 claims extracted, 25 adversarially verified by 3-vote panels (19 confirmed, 6 refuted) — then cross-audited against an independent OpenAI Codex analysis of the same question (§5), whose three key papers were verified by direct search and folded in (§1.12). Sections 1–2 are verified; Section 3 is analysis/synthesis (flagged); Section 4 is the proposed design.

---

## 0. Executive summary

The verified literature converges on this: **a from-first-principles agent memory engine should NOT be a vector database.** It should be a token-level (externally stored, human-readable) **hybrid system** with:

1. A **three-operation API** — write / manage / read — not a single `retrieve()` call
2. **LLM-driven belief revision at write time** (ADD / UPDATE / INVALIDATE / NOOP)
3. **Bitemporal validity** on facts (contradicted facts are invalidated, never deleted)
4. A **typed-substore router** — because adversarial benchmarking shows *no single architecture dominates*: graph memory both wins (Zep on LongMemEval; HippoRAG's 10–30× cheaper single-step multi-hop) and loses (Mem0-g underperforms dense memory on multi-hop; StructMem collapses on coexisting-fact retrieval) depending on task shape
5. A **tiered read path**: LLM-free activation spreading (PPR × ACT-R decay) for cheap queries, escalating to MRAgent-style **active reconstruction** (bounded LLM-guided graph exploration) for hard ones — active retrieval is provably more expressive than passive top-k (ICLR 2026)
6. **Provenance always**: the ledger already exists — beater.js journals every agent step to SQLite, and beater-agents has canonical `memory.read`/`memory.write` span kinds. Memory is a *projection over traces you already capture*, not a new storage system.

Two verified facts make this the right project for a small Rust team:

- **Current systems are bound by architecture, not model intelligence or context budget.** Scaling retrieval k (4→20) or swapping in stronger backbone LLMs yields little and sometimes degrades results (arXiv 2605.26667). Architecture is the open lever.
- **The cross-field territory (SDM, VSA, Hopfield, ACT-R spreading activation, event sourcing, MDL) has no verified agent-memory implementation on arXiv.** The most first-principles ideas are genuinely unclaimed.

---

## 1. Verified findings (survived 3-vote adversarial verification)

### 1.1 The design space is mapped — position against it
**Confidence: high.** Three independent 2025–26 surveys converge:
- **arXiv 2512.13564** ("Memory in the Age of AI Agents", Dec 2025): three dominant realizations — **token-level** (external text/JSON/graph; where a Rust engine sits), **parametric** (weight-encoded), **latent** (KV-cache-as-memory). Proposes factual / experiential / working taxonomy because long/short-term "has proven insufficient."
- **arXiv 2504.15965** (Huawei Noah's Ark): 3D-8Q taxonomy — object (personal/system) × form (non-parametric/parametric) × time (short/long).
- **arXiv 2602.19320** ("Anatomy of Agentic Memory"): four structural families of deployed systems — lightweight semantic (flat vector top-k), entity-centric, episodic/reflective, structured/hierarchical (MemGPT, Zep, MAGMA) — with failure modes mapping onto structure.

### 1.2 Memory is a first-class primitive with a three-op API
**Confidence: high.** arXiv 2512.13564 + 2404.13501 (ACM TOIS): memory ≠ RAG ≠ context engineering. Canonical decomposition:
- **write** — transform raw observations into concise stored content
- **manage** — summarize into higher-level concepts, merge redundancy, forget irrelevant items (offline)
- **read** — select information to support the next action

This is the direct API blueprint. (Contested by one paper — 2604.11628 "Back to Basics" argues retrieval+generation suffices — noted for honesty.)

### 1.3 Build token-level; treat parametric/KV-latent as research bets
**Confidence: medium** (single source, 2-1 vote). arXiv 2404.13501: parametric memory blocked on meta-training cost and collateral damage to unrelated memories. Textual/token-level is the dominant deployed form.

### 1.4 Graphs do NOT automatically fix multi-hop — the core surprise
**Confidence: high.** Two against-interest results:
- Mem0's own paper (**arXiv 2504.19413**, ECAI 2025): graph variant Mem0-g scores *lower* on multi-hop (J=47.19) than dense natural-language Mem0 (J=51.15) — "inefficiencies or redundancies in structured graph representations." (Mem0-g does win on temporal/open-domain.)
- MemFail (**arXiv 2605.26667**, Dawn Song's group): graph-based StructMem excels at causal/multi-hop but *collapses* on retrieving multiple coexisting facts; vector-based Mem0 shows the inverse.

### 1.5 No architecture dominates → hybrid mixture-of-memories with routing
**Confidence: high** (that no system dominates); the routing fix itself is *recommended but unproven*. MemFail: "no single system dominates: each architecture exhibits a distinctive failure signature... hybrid systems that route memories to the appropriate substore could combine these strengths." **No published system ships this.** This is the buildable, novel-ish target.

### 1.6 Architecture is the bottleneck, not model or k
**Confidence: high.** MemFail: scaling k∈{4..20} or backbone strength "yields little improvement, and in several cases degrades performance." Also: verbose memories **pollute embedding space** on retrieval-bottlenecked tasks (accuracy *declines* with stored tokens on Coexisting-Facts) — embedding brittleness is real and task-dependent.

### 1.7 Write-time belief revision is proven and copyable
**Confidence: high.**
- **Mem0** (2504.19413 + live code): two-phase LLM write path — fact extraction, then the LLM itself (function-calling, no classifier) picks ADD/UPDATE/DELETE/NOOP per fact against retrieved similar memories.
- **Zep/Graphiti** (**arXiv 2501.13956**): temporally-aware knowledge graph; facts carry validity windows; contradicted facts are **invalidated bitemporally, not deleted** — history is preserved and temporally queryable.
- Combine both: LLM-adjudicated writes + bitemporal validity = auditable belief revision.

### 1.8 Single-shot graph retrieval can pay for itself on tokens
**Confidence: medium** (vendor-authored numbers; benchmark-shaped wins).
- **HippoRAG** (**arXiv 2405.14831**, NeurIPS 2024): LLM OpenIE graph + Personalized PageRank ≈ iterative retrieval (IRCoT) quality at **10–30× cheaper, 6–13× faster** query time. Caveat: *excludes offline indexing cost*; loses to plain ColBERTv2 on HotpotQA R@2.
- **Zep** (2501.13956): 71.2% vs 60.2% full-context baseline on LongMemEval-S at ~2.6s vs ~29s. Superseded by 2026 systems (86–95%) but the structural point stands: *returning a graph-neighborhood answer in one step beats multi-round LLM retrieval loops on tokens.*

### 1.9 Write amplification and read latency are first-order design axes
**Confidence: high.** arXiv 2602.19320 Table 5 ("Agency Tax"): index construction varies ~5× (Nemori 7.04M tokens vs SimpleMem 1.3M; A-MEM took 15 *hours* offline); read latency varies ~30× (SimpleMem 1.06s vs MemoryOS 32.4s — vs 1.73s for brute-force context stuffing). Some "sophisticated" systems are slower than stuffing. **Report tokens-per-stored-memory and read latency as headline metrics.**

### 1.10 Structured writes fail silently on weaker backbones
**Confidence: high.** arXiv 2602.19320 §4.4: entity extraction / relation construction / dedup produce invalid structured output that **corrupts long-term memory while the agent converses fluently**. Nemori: 0.781 accuracy (17.9% format errors) on gpt-4o-mini → 0.447 (30.4% format errors) on Qwen-2.5-3B. Append-only systems are robust. → The engine needs **constrained decoding, schema validation, and write-rejection/repair** between LLM and store.

### 1.11 Evaluation methodology
**Confidence: high.** arXiv 2602.19320:
- Most benchmarks are theoretically saturated (fit in modern context windows). Only benchmarks structurally exceeding the window (**LongMemEval-M, >1M tokens**) genuinely require external memory. Always report the **Context Saturation Gap** (Δ vs brute-force full-context baseline).
- **Never use F1/BLEU** — lexical overlap misranks memory systems ("Paraphrase Penalty"; A-MEM ranks 4th under LLM judge, 5th under F1 at 0.116). Use calibrated LLM judges; rankings are robust across rubrics.
- LongMemEval original paper: **arXiv 2410.10813**. LoCoMo: **arXiv 2402.17753** (contested; LoCoMo-Plus 2602.10715 exists).

### 1.12 Post-audit additions (from the Codex cross-audit; existence + abstracts verified by direct search, not the 3-vote panel)
- **MRAgent — "Memory is Reconstructed, Not Retrieved" ([arXiv 2606.06036](https://arxiv.org/abs/2606.06036), [ICLR 2026](https://openreview.net/forum?id=YPoHy6lgKP)):** memory as a **Cue → Tag → Content associative graph** plus an **active reconstruction** read path — the LLM iteratively explores and prunes retrieval paths as evidence accumulates, instead of static retrieve-then-reason. Includes a theoretical proof that **active retrieval policies are strictly more expressive than passive retrieval**. [Press coverage](https://venturebeat.com/orchestration/new-agentic-memory-framework-uses-118k-tokens-per-query-langmem-burns-through-3-26m) reports ~118k tokens/query vs LangMem's 3.26M (~27×) — treat the exact ratio as unverified, the mechanism as peer-reviewed. **This is the strongest single paper for the read path.**
- **LongMemEval-V2 ([arXiv 2605.12493](https://arxiv.org/abs/2605.12493), May 2026):** 451 curated questions over web-agent *trajectories* (up to 500 trajectories / 115M tokens; WebArena+WorkArena environments) testing five abilities: **static state recall, dynamic state tracking, workflow knowledge, environment gotchas, premise awareness**. This is the benchmark that matches real agent workloads (vs conversational QA) — it directly answers our open question #4 and its five abilities read like a spec for our typed substores.
- **StructMemEval — "Evaluating Memory Structure in LLM Agents" ([arXiv 2602.11243](https://arxiv.org/abs/2602.11243)):** tasks humans solve by *organizing* knowledge (transaction ledgers, to-do lists, trees). **Retrieval-only systems cannot solve them**; flexible agentic memory does better but is far from perfect; failures trace to suboptimal memory organization. Independent confirmation that memory needs typed structure, not chunks.
- To read: **Eywa — provenance-grounded long-term memory ([arXiv 2605.30771](https://arxiv.org/pdf/2605.30771))** — closest published work to our provenance-first design; check for overlap before claiming novelty.

---

## 2. Refuted claims — do NOT cite these (they circulate widely)

| Refuted claim | Vote |
|---|---|
| Mem0's headline "26% over OpenAI memory, 91% latency reduction, >90% token savings" | 1-2 |
| "Zep's graph store is ~600k tokens for a 26k-token transcript" (write-amplification factoid) | 0-3 |
| Zep beats MemGPT on DMR 94.8% vs 93.4% (comparison methodology disputed) | 0-3 |
| HippoRAG "outperforms SOTA RAG by up to 20%" (the headline; the mechanism results in §1.8 stand) | 1-2 |
| Clean episodic=records / semantic=weights / procedural=patterns mapping (2504.15965 doesn't actually validate it) | 0-3 |
| "Graphs are better at relationships but fail at general retrieval" as a clean dichotomy | 1-2 |

Also: Mem0 and Zep numbers are mutually contested (Zep disputed Mem0's baselines; Mem0 corrected Zep's LoCoMo claim 84%→58.44%). Architectural descriptions survive; headline numbers don't.

---

## 3. The open territory (UNVERIFIED — my synthesis; check IDs before citing)

The verification pass produced **zero confirmed claims** for cross-field mechanisms — nobody has validated these as LLM-agent memory on arXiv, or the evidence didn't survive. Either way: **this is the novelty budget.**

- **ACT-R declarative memory (Anderson):** base-level activation `B_i = ln(Σ t_j^{-d})` — recency × frequency power-law decay — plus spreading activation from context, partial matching, retrieval threshold. **Pure math, no LLM, trivially implementable in Rust as the ranking function.** CoALA (arXiv 2309.02427) frames this but doesn't build it.
- **Personalized PageRank as spreading activation:** HippoRAG proved PPR works for single-step multi-hop (§1.8). ACT-R decay + PPR over a fact graph = principled, LLM-free read tier.
- **Kanerva's Sparse Distributed Memory (1988):** content-addressable, distributed writes, graceful degradation. Borrowable as sparse binary sketches for approximate content addressing.
- **Vector Symbolic Architectures / hyperdimensional computing** (Kleyko surveys: arXiv 2111.06077, 2111.05498): **binding** (XOR/circular convolution), **superposition**, **permutation** give *algebraic* compositional queries — role-filler unbinding answers "user prefers ?" without graph traversal. Noise-tolerant, SIMD-fast in Rust. Unproven for agent memory at scale.
- **Modern Hopfield networks** (arXiv 2008.02217): exponential-capacity associative retrieval ≈ attention; the useful bit is *iterative pattern completion* — retrieve, re-query with the retrieved pattern, converge. (MRAgent's active reconstruction is arguably the LLM-flavored version of this.)
- **Complementary Learning Systems (McClelland et al.):** fast instance store + slow semantic store + replay-based consolidation = the write/manage split; "sleep-time compute" (Letta, arXiv ~2504.12717) is its engineering twin.
- **Surprise-based event segmentation** (EM-LLM, arXiv 2407.09450): segment episodes at Bayesian-surprise boundaries instead of fixed-size chunks.
- **Event sourcing + bitemporal databases (XTDB lineage):** append-only immutable log as source of truth; all indexes are rebuildable projections; valid-time + transaction-time on every fact. Composes perfectly with §1.7's invalidate-don't-delete.
- **MDL / compression-as-understanding:** consolidation = lossy compression with provenance back to the lossless log; compression ratio as a memory-quality metric.
- **Spaced repetition / Ebbinghaus decay** (MemoryBank, arXiv 2305.10250): decay scores schedule eviction *and* re-consolidation.
- **Information foraging theory (Pirolli & Card):** retrieval as expected-information-gain per token spent — the theory behind a token-budgeted, progressive-disclosure read API and `suggested_next_queries` ("memory scent").

---

## 4. Proposed design: `beater.memory` (working name: **engram**)

Composes the verified findings, the unclaimed mechanisms, and the audit's repo grounding. Rust crate, embedded in beater.js next to beater.agents.

### Core principle
**The memory is a projection over an append-only bitemporal ledger — and the ledger already exists.** beater.js journals every agent/LLM/tool step to SQLite; beater-agents defines canonical `memory.read`/`memory.write` span kinds (`beater-schema`), CanonicalSpans, and a Tantivy full-text crate (`beater-search`). Don't build new storage; build the **projector** and the **query engine**. No LLM in the write hot path (§1.10: append-only is robust). LLM at read time only when escalated and budgeted (§1.12 MRAgent). LLM consolidation runs off-path at "sleep time" and is re-runnable because the ledger is lossless.

### Layers

1. **Ledger (write).** The existing journal/span forest. Observations append with transaction time. Write amplification ≈ 0 at write time; crash-safe by construction.

2. **Distiller (manage, offline/sleep-time).** LLM pass over new ledger segments:
   - extract concise facts (entity–relation–object + free-text form, dual-encoded)
   - adjudicate against retrieved neighbors: **ADD / UPDATE / INVALIDATE / NOOP** (§1.7) — INVALIDATE sets `valid_to`, never deletes (bitemporal, Graphiti-style)
   - **constrained decoding + schema validation + repair loop** — mandatory per §1.10
   - every node carries provenance pointers to ledger spans
   - surprise-segmented episode boundaries (EM-LLM) instead of fixed chunks

3. **Typed graph + substores (projections).** Node types (merging our taxonomy with the audit's, which maps 1:1 onto LongMemEval-V2's five abilities):
   `Episode` (span-backed event) · `Fact` (bitemporal semantic) · `Entity/Cue` (tool, route, file, error, model, user) · `Tag` (relation-shaped bridge, MRAgent-style) · `Procedure` (runbook/workflow) · `State` (remembered environment condition) · `Gotcha` (recurring failure mode) · `AntiMemory` (looked relevant before, was misleading) · `Topic` (cluster/summary)
   Edge types: `mentions, caused_by, fixes, contradicts, supersedes, before/after, part_of, derived_from, blocks, enables, observed_in`.
   **Contradictions are edges, never collapsed into a summary.** Storage-wise this is the MemFail mixture-of-memories (§1.5): facts in an embedding index over *concise* text (§1.6 pollution), relations in adjacency tables, episodes in the span store, procedures as structured docs. Routing starts as type-based rules; LLM adjudication only on ambiguity. StructMemEval (§1.12) independently confirms typed structure is load-bearing, not decoration.

4. **Tiered read path.**
   - **Tier 0 — cue seeding (no embeddings needed for MVP):** Tantivy lexical match + entity extraction seed the graph. Vector ANN (HNSW, [arXiv 1603.09320](https://arxiv.org/abs/1603.09320)) added later as *another seed channel, not the system*.
   - **Tier 1 — activation (no LLM):** Personalized PageRank from seeds (HippoRAG mechanism) blended with ACT-R base-level activation (recency × frequency decay), traversing typed edges. Returns a ranked evidence bundle. Rust-fast, token-free.
   - **Tier 2 — active reconstruction (budgeted LLM):** when Tier 1 confidence is low or the query is compositional, an MRAgent-style loop explores/prunes graph paths with the LLM, spending against an explicit budget. Provably more expressive than passive retrieval (2606.06036); the escalation policy keeps its cost exceptional rather than default.

   API shape (adopted from the audit, extended):
   ```rust
   MemoryQuery  { question, scope, max_tokens, as_of: Option<Time>, require_fresh: bool,
                  modes: [semantic, episodic, procedural, gotcha, state] }
   MemoryAnswer { answer, evidence, cited_spans, contradictions, stale_assumptions,
                  suggested_next_queries, token_estimate, tier_used }
   ```
   Memory answers questions; it does not return documents. `as_of` makes temporal queries native — the query class flat vector RAG cannot express. `stale_assumptions` = premise awareness (LME-V2 ability #5). `suggested_next_queries` = foraging scent.

### What's genuinely novel-ish here
Bitemporal event-sourced core *reusing the agent's own journal* + typed substore routing + anti-memory/contradiction surfacing + a tiered read path (LLM-free activation → budgeted active reconstruction) + token-budgeted answer-first API. Each piece has precedent; the composition ships nowhere (§1.5 verified; check Eywa 2605.30771 for provenance overlap).

### Evaluation plan
- **Headline: LongMemEval-V2** (2605.12493) — agent-trajectory memory, five abilities that map onto our node types; LME-V2-Small (100-trajectory shared haystack) is the tractable starting split
- **StructMemEval** (2602.11243) — the router/typed-structure ablation target
- **LongMemEval-M** (>1M tokens, structurally requires external memory) + LoCoMo with caveats; replicate 2602.19320's harness
- Report: **Context Saturation Gap** (Δ vs full-context stuffing), calibrated-LLM-judge accuracy (never F1), **write tokens per stored memory**, **read latency per tier**, tokens-into-context per answer
- Ablate: router on/off; Tier 1 (PPR/decay) on/off; Tier 2 escalation on/off; verbose-vs-concise fact storage (should reproduce embedding pollution)

### Open questions
1. Do SDM/VSA/Hopfield mechanisms hold up as real agent memory? (Unclaimed territory — highest novelty upside.)
2. True TCO of graph memory once offline indexing is counted (HippoRAG's 10–30× excludes it; our sleep-time distiller must be metered).
3. What routing policy wins — rules, learned classifier, or LLM adjudication — and does routing overhead eat the gains?
4. What Tier 1→Tier 2 escalation policy (confidence threshold? query-shape classifier?) preserves MRAgent-level quality at a fraction of its cost?
5. How much does Eywa (2605.30771) overlap with the provenance design?

---

## 5. Cross-audit: Codex's proposal vs this report

Jaden ran the same prompt through OpenAI Codex; its answer was folded in above. Scorecard:

**Codex got right (adopted):** grounding in the actual repo (memory.read/write span kinds, Tantivy, CanonicalSpans — all verified real here); memory-as-projection-over-traces; MRAgent/LongMemEval-V2/StructMemEval (all real, all missed by our sweep — verified by direct search); anti-memory, Gotcha/State node types, contradiction edges, premise awareness; MemoryQuery/MemoryAnswer with `token_estimate` and `suggested_next_queries`; lexical-seed-first MVP with embeddings deferred; "vector search is one sensory organ, not the system."

**This report adds (Codex lacked):** adversarial verification and a refuted-claims list (Codex repeated HippoRAG's 10–30× without the offline-indexing caveat; would happily have cited the refuted Mem0/Zep headline numbers); the write path — belief-revision op-set, bitemporal validity, and the *silent-corruption-on-weak-backbones* finding that mandates constrained decoding (§1.10); write-amplification economics (§1.9); embedding-pollution (§1.6); the no-architecture-dominates evidence that justifies typed substores as *storage* not just node labels (§1.4–1.5); eval methodology (saturation gap, LLM-judge vs F1); the unclaimed cross-field territory (§3); ACT-R decay + PPR as the LLM-free ranking tier.

**Synthesis judgment:** Codex's best structural idea (reconstruction over a typed provenance graph) and this report's best structural idea (tiered, budgeted, bitemporal, verified-economics engine) are complementary — the merged design in §4 is stronger than either original.

---

## Sources (verified-primary, fetched by the workflow)

Surveys: 2512.13564 · 2504.15965 · 2404.13501 · 2602.19320 ("Anatomy of Agentic Memory", the eval-methodology anchor)
Systems: 2504.19413 (Mem0) · 2501.13956 (Zep/Graphiti) · 2405.14831 (HippoRAG) · 2401.18059 (RAPTOR) · 2502.14802 · 2601.03417 · 2606.06036 (MRAgent) · 2605.30771 (Eywa, unread)
Adversarial: 2605.26667 (MemFail) · 2604.20943 · 2605.06527 · 2504.13171
Benchmarks: 2410.10813 (LongMemEval) · 2605.12493 (LongMemEval-V2) · 2602.11243 (StructMemEval) · 2402.17753 (LoCoMo) · 2510.18866 · 2307.03172 (Lost in the Middle)
Cross-field: 2111.06077 + 2111.05498 (VSA surveys) · 2601.02744 · 1603.09320 (HNSW)

*Caveats: 2602.19320 and 2605.26667 are non-peer-reviewed 2026 preprints; MemFail uses partly LLM-synthetic data with GPT-5-mini as subject and grader; Mem0/Zep numbers are vendor-reported and mutually contested; §1.12 items verified for existence/abstract only; this field moves monthly.*
