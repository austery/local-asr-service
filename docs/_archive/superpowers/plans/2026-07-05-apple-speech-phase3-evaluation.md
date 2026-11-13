# Apple Speech Phase 3 Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SPEC-014 Phase 3 evaluation reproducible enough to decide whether `apple-speech` is usable through the normal API, which workflows it should be recommended for, and where it must remain ASR-only or experimental.

**Architecture:** Keep evaluation at the gateway boundary. The harness calls `/v1/audio/transcriptions` with explicit `model` and `language`, records response shape, timing/SRT behavior, runtime duration, optional process-tree RSS, and enough text summary to compare Apple Speech against Paraformer and Qwen3-ASR without turning this repo into a general ASR leaderboard.

**Tech Stack:** Python 3.11, FastAPI contract tests, pytest, requests, POSIX `ps` sampling, `uv`.

---

## Context Alignment

Previous model evaluation already exists and sets the policy:

- `docs/SPEC-008-Dynamic-Model-Switching.md`: two-pass benchmarks became scenario recommendations.
- `docs/SPEC-013-Production-Decoupled-Diarization-Pipeline.md`: `qwen3-sortformer` stayed reachable for evaluation but was rejected as a production recommendation after real meeting probes showed weak speaker-segment coverage, fragmentation, and high memory cost.
- `docs/ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md`: new model work is evaluation-first; this repo owns API compatibility, queueing, resource isolation, model registry, response normalization, and probe scripts.
- `MODELS.md`: active model recommendation surface and historical evaluation record.

Phase 3 therefore answers one question:

```text
After Apple Speech is connected to the existing pipeline, is it actually usable through the normal API, and if yes, for which scenario?
```

## File Structure

- Modify `src/api/routes.py`: preserve JSON `segments=[]` when the service returns an empty segment list.
- Add `benchmarks/phase3_evaluation.py`: typed Phase 3 HTTP harness.
- Add `tests/unit/test_phase3_evaluation.py`: pure tests for response-shape analysis, SRT validation, request payload construction, and memory parser helpers.
- Modify `tests/integration/test_model_api.py`: canary for `segments=[]` JSON contract.
- Modify `docs/SPEC-014-Apple-SpeechAnalyzer-Integration.md`: define Phase 3 as a usability/recommendation gate.
- Modify `MODELS.md`: clarify Apple Speech recommendation status and benchmark recency.

---

### Task 1: Lock JSON Empty-Segments Contract

**Files:**
- Modify: `tests/integration/test_model_api.py`
- Modify: `src/api/routes.py`

- [ ] **Step 1: Write the failing API canary**

Add a test asserting `output_format=json` preserves `segments=[]` when the service returns an empty list.

- [ ] **Step 2: Run the canary and verify it fails**

Run:

```bash
uv run python -m pytest -p no:tach tests/integration/test_model_api.py -q -k preserve_empty_segments
```

Expected: FAIL because the route currently initializes `segments=None` when the list is empty.

- [ ] **Step 3: Preserve list semantics in the route**

Change the route so `effective_format == "json"` and `segments_obj` being a list always returns a list, including `[]`.

- [ ] **Step 4: Verify the canary passes**

Run:

```bash
uv run python -m pytest -p no:tach tests/integration/test_model_api.py -q -k "preserve_empty_segments or apple_speech"
```

Expected: PASS.

---

### Task 2: Add Phase 3 Evaluation Summary Primitives

**Files:**
- Add: `tests/unit/test_phase3_evaluation.py`
- Add: `benchmarks/phase3_evaluation.py`

- [ ] **Step 1: Write failing tests for `analyze_segments`, `summarize_json_response`, `summarize_srt_text`, `build_request_data`, and `parse_ps_rss_kb`.**
- [ ] **Step 2: Run `uv run python -m pytest -p no:tach tests/unit/test_phase3_evaluation.py -q` and verify failure.**
- [ ] **Step 3: Implement typed dataclasses and pure helpers.**
- [ ] **Step 4: Re-run the focused tests and verify pass.**

---

### Task 3: Add HTTP Probe Runner and Optional Resource Sampling

**Files:**
- Modify: `benchmarks/phase3_evaluation.py`
- Modify: `tests/unit/test_phase3_evaluation.py`

- [ ] **Step 1: Add failing tests for `build_probe_result`, invalid RSS parsing, and unavailable-process sampling.**
- [ ] **Step 2: Verify tests fail.**
- [ ] **Step 3: Implement `ProbeResult`, process-tree RSS sampling, `PeakRssSampler`, `run_json_probe`, optional `run_srt_probe`, and CLI.**
- [ ] **Step 4: Verify focused tests pass.**

Example command:

```bash
uv run python benchmarks/phase3_evaluation.py \
  --file /path/to/audio.wav \
  --language zh-CN \
  --models apple-speech paraformer qwen3-asr \
  --base-url http://localhost:50700 \
  --save \
  --server-pid <pid> \
  --srt-probe
```

---

### Task 4: Sync SPEC-014 and MODELS Evaluation Language

**Files:**
- Modify: `docs/SPEC-014-Apple-SpeechAnalyzer-Integration.md`
- Modify: `MODELS.md`

- [ ] **Step 1: Update SPEC-014 Phase 3 to say it is a usability/recommendation gate, not a leaderboard.**
- [ ] **Step 2: Add Phase 3 harness commands and required evidence fields.**
- [ ] **Step 3: Update MODELS.md to mark Apple Speech recommendation strength as pending Phase 3 evidence and note older benchmarks predate Apple Speech.**
- [ ] **Step 4: Search for conflicting Apple Speech production/diarization wording.**

---

### Task 5: Verification

- [ ] Run focused tests:

```bash
uv run python -m pytest -p no:tach tests/unit/test_phase3_evaluation.py tests/integration/test_model_api.py -q
```

- [ ] Run baseline:

```bash
uv run python -m pytest tests/unit tests/integration
```

- [ ] Run lint:

```bash
uv run ruff check benchmarks/phase3_evaluation.py tests/unit/test_phase3_evaluation.py tests/integration/test_model_api.py src/api/routes.py
```

- [ ] Inspect diff:

```bash
git diff -- docs/superpowers/plans/2026-07-05-apple-speech-phase3-evaluation.md benchmarks/phase3_evaluation.py tests/unit/test_phase3_evaluation.py tests/integration/test_model_api.py src/api/routes.py docs/SPEC-014-Apple-SpeechAnalyzer-Integration.md MODELS.md
```

## Self-Review

- Spec coverage: Phase 3 comparisons, timing/SRT, runtime duration, peak memory, failure rate, and recommendation outcome are covered.
- Placeholder scan: no TODO/TBD placeholders.
- Scope: Phase 4 dictation vocabulary and Phase 5 diarization remain out of scope.
- Type consistency: new benchmark code uses typed dataclasses and `Mapping[str, object]`, not `Any`.
