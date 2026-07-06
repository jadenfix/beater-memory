---
name: review-pr
description: High-recall, high-precision independent review of a beater-memory PR. Use when asked to review a PR in jadenfix/beater-memory (e.g. "/review-pr 45"). Reviews must be done by an agent that did NOT author the PR.
---

# beater-memory PR review

You are an independent, non-author reviewer for `jadenfix/beater-memory`. The argument is a PR number: `$ARGUMENTS`. Several agents work this repo concurrently — assume nothing about freshness, and never rubber-stamp. This rubric teaches you *how* to find bugs on any PR; it is deliberately not a list of past bugs to grep for.

## Ground rules

- **Non-author only.** Check `gh pr view <N> -R jadenfix/beater-memory --json commits -q '.commits[].messageHeadline'` — if you recognize any commit as your own work from this session, stop and hand the review to another agent.
- Read-only: do not modify the main clone, do not run `cargo` in a directory another agent may be building in. CI already builds per-PR; review by reading.
- Precision: every **blocker** carries a concrete traced failure scenario (specific input/state → specific wrong behavior, with `file:line`). If you cannot trace one, it is a nit.
- Recall: read the ENTIRE diff, the referenced issues, and the surrounding code of every touched file in the PR head or merge result. Also compare default-branch context for supersession and stale assumptions. Bugs live at the seams the diff doesn't show.
- Approval gate: `APPROVE` is allowed only when there are zero blockers, the PR is open, non-draft, mergeable at the inspected head SHA, and every required check in the current check rollup is completed successfully for that head. Pending, failed, stale, or missing checks are a hold, not an approval.

## Procedure

1. `gh pr view <N> -R jadenfix/beater-memory --json title,body,author,state,isDraft,headRefOid,baseRefOid,files,mergeStateStatus,statusCheckRollup`
2. `gh pr diff <N> -R jadenfix/beater-memory` — all of it.
3. `gh issue view <issue> -R jadenfix/beater-memory` for every referenced issue; the issue defines the intended scope.
4. **Supersession check:** `git log` on the default branch plus targeted `git log -p` on touched files — an equivalent fix may already have landed → REJECT (superseded).
5. **Freshness check:** after any wait, force-push, PR body edit, or CI rerun, re-read PR state, head SHA, base SHA, check rollup, and linked issue state.
6. **Gate check:** record the inspected `headRefOid` in the review. If checks are pending, failed, stale, or absent for that head, do not approve; post a neutral hold comment instead.
7. **Overlap check:** `gh pr list -R jadenfix/beater-memory --state open --json number,title,headRefName,baseRefName,files` — flag open PRs touching the same paths and whether merge order matters. If the list output is too broad or truncated, run `gh pr view <other> -R jadenfix/beater-memory --json files` for candidates.
8. Hunt for bugs using the method below.
9. Post the review (format at the bottom) and return a structured verdict.

## How to find bugs (do this — don't just tick boxes)

- **Trace one path end to end.** Follow one event from ledger ingest through distillation, projection, and a query answer — into the rejection, repair, and rollback branches, not just the happy path.
- **Review from three seats.** beater-memory serves a **querying agent** (it acts on recalled facts — a wrong, stale, or provenance-free answer becomes a wrong action), a **distillation pipeline** (provider output is untrusted JSON that arrives malformed, oversized, or slow), and an **operator** (ledger integrity, projection rebuilds, backup/restore). For the code in the diff, ask how it hurts each of the three.
- **Enumerate failure modes** for every new input, call, or state transition: empty · malformed · oversized · slow/hung · repeated/retried · concurrent · out-of-order · partial failure · adversarial/untrusted.
- **Follow the seams the diff hides:** callers of changed signatures, callees now leaned on, invariants elsewhere that assumed the old behavior.
- **Reverted-fix test:** would any test in the PR still pass if the fix were reverted? If yes, it proves nothing — a blocker for a bugfix PR.
- **Adversarially verify** each candidate blocker: try to refute it against the code. Survives → blocker. No concrete trace → nit.
- **Preserve durable lessons** under `Durable guidance`; a follow-up author lands accepted guidance in both repo agent guidance (`AGENTS.md` where applicable) and this file from a separate worktree/PR.

## What to look for (general bug classes)

Correctness & honesty of the contract:
- [ ] Return values and status flags tell the caller the truth — failures and no-ops are never reported as success; anything dropped, truncated, or budget-cut says so, so the consumer can distinguish "absent" from "omitted."
- [ ] Docs and declared schemas match what the code actually does — no present-tense claims for a stub.

Resource, lifecycle & availability:
- [ ] Everything that can grow is bounded: import batches, graph traversal steps, repair attempts, caches, response sizes. Unbounded growth on external input is a blocker; a check after full allocation is not a bound.
- [ ] Every provider/subprocess call has a timeout **and** a bounded retry story; cleanup runs on all exit paths.
- [ ] SQLite transactions are short; no long-running work (provider calls, network) inside a write transaction.

Tests:
- [ ] Tests exercise the actual failure mode (survive the reverted-fix question); budgets/limits tested at, below, above the boundary.

Fit & simplicity:
- [ ] The change does exactly what its issue needs — no speculative abstraction, dead branch, or unused knob.
- [ ] It fits ARCHITECTURE.md and `research/agent-memory.md`: memory is a projection over append-only traces, not a vector store with ad-hoc writes.

## beater-memory-specific bug classes (check every one the diff touches)

Ledger & projection integrity (the append-only spine):
- [ ] The ledger is append-only: no code path mutates or deletes ingested events. Projections must be rebuildable from the ledger alone; any state that exists only in the projection is a design smell, and a blocker if answers depend on it.
- [ ] **No provider calls inside projection write transactions** — this is a standing hard rule. Any model/LLM/command invocation while a write transaction is open is a blocker.
- [ ] Imports are atomic: a malformed row rolls back the whole batch and reports the bad source row; partial ingestion of a batch is a blocker.
- [ ] Ledger validation rejects malformed events before they enter the ledger — rejection is counted in metrics, never a silent drop.

Bitemporal correctness (the two clocks are different):
- [ ] `as_of_unix_ms` (observed validity) and `known_at_unix_ms` (ledger transaction time) are never conflated. A query with `known_at` must not see facts, edges, or invalidations the ledger ingested later — including relation edges, whose provenance carries the source event.
- [ ] `INVALIDATE` and anti-memory affect reads only within the correct temporal window; an invalidated fact must not resurface via a different substore or traversal path.

Distillation & provider boundary (untrusted JSON):
- [ ] Provider output goes through strict schema parsing with bounded repair attempts; exceeding the bound rejects with metrics, never loops or half-applies.
- [ ] Distiller decisions (`ADD/UPDATE/INVALIDATE/NOOP`) are validated against the graph before projection writes; a decision referencing a nonexistent node fails the decision, not the store.

Answer quality (the agent acts on this):
- [ ] Every answer carries provenance to source events; contradiction warnings are surfaced, not silently resolved to one side.
- [ ] Token budgets are enforced and truncation is labeled; tier routing (typed substores → graph activation → active reconstruction) falls back conservatively, and Tier 2 exploration respects its step and token bounds.
- [ ] Idempotency keys on direct `remember` writes actually dedupe on retry.

Operations:
- [ ] HTTP API paths stay authenticated; backup/restore and integrity checks cannot corrupt a live store (locking or exclusivity is explicit); orphan repair never deletes reachable data.

## Verdict & posting

Post exactly one review. Make the GitHub review action match the verdict:

```
gh pr review <N> -R jadenfix/beater-memory --approve --body "<body>"          # APPROVE only
gh pr review <N> -R jadenfix/beater-memory --request-changes --body "<body>"  # REQUEST-CHANGES or REJECT
gh pr review <N> -R jadenfix/beater-memory --comment --body "<body>"          # HOLD only
```

Body format — first line is the verdict, nothing above it:

```
VERDICT: APPROVE | REQUEST-CHANGES | HOLD (checks-pending | checks-failed | stale | draft | not-mergeable) | REJECT (superseded | wrong-approach)

<one-paragraph summary: what the PR does, whether it fixes the traced failure>

Blockers:
- <file:line — traced failure scenario>   (or "none")

Nits:
- <file:line — suggestion>                (or "none")

Durable guidance: <candidate reusable invariant for follow-up docs, or "none">

Overlap: <open PRs touching same paths + merge-order note, or "none">

Inspected head: <headRefOid>; base: <baseRefOid>; checks: <passing | pending | failed | stale | missing>

— independent review agent (non-author)
```

APPROVE only with zero blockers and fresh passing gates on the inspected head. REQUEST-CHANGES when fixable blockers exist. HOLD when the code review is clean but CI, mergeability, freshness, or draft state prevents approval. REJECT when superseded or the approach conflicts with the projection-over-ledger architecture. Do not merge — merging is the coordinator's job after CI + mergeability recheck.

## Deep mode (optional)

If asked for a "deep" review, fan out three parallel non-author subagents with distinct lenses — (a) ledger/bitemporal correctness, (b) provider-boundary robustness, (c) scope/over-engineering — then adversarially verify each candidate blocker yourself before posting.
