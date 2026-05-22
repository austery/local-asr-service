# Qwen3 Sortformer Documentation Repositioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reposition `qwen3-sortformer` in repository and Research Notes documentation as an experimental reachable deletion candidate without changing runtime code.

**Architecture:** This is a documentation-only alignment pass. Repo-facing docs will express the public model stance, ADR/SPEC docs will record the decision and evidence, and Research Notes will preserve the personal project lesson and future deletion criteria.

**Tech Stack:** Markdown, repo ADR/SPEC conventions, Obsidian Research Notes markdown.

---

## File Map

- Modify `README.md` for public-facing model selection and reachable-experiment wording.
- Modify `MODELS.md` for model table, selection guide, and resource/quality tradeoff notes.
- Modify `docs/ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md` for experimental reachability and deletion posture.
- Modify `docs/SPEC-013-Production-Decoupled-Diarization-Pipeline.md` for the 2026-05-20 real 1:1 reassessment.
- Optionally modify `docs/superpowers/2026-05-19-qwen3-sortformer-longform-defects.md` only to point from earlier probe evidence to the newer reassessment.
- Modify Research Notes MOC `03-project/2026-03_local-asr-service-MOC.md` to remove stale enablement language.
- Create one Research Notes reassessment note in `01-inbox/` for the real 1:1 probe and the decision to stop deeper local pipeline investment.

### Task 1: Public Repo Positioning

**Files:**
- Modify: `README.md`
- Modify: `MODELS.md`

- [ ] **Step 1: Update README use-case and model descriptions**

  Change the `qwen3-sortformer` wording from production-adjacent batch-path
  language to explicit experimental evaluation language. Keep `qwen3-asr` as
  the English text-quality path and `paraformer` as the existing
  diarization-capable path with English quality limits.

- [ ] **Step 2: Update the model reference**

  In `MODELS.md`, state that requestable only means the experiment remains
  callable. Record the latest probe conclusion: Qwen3 text quality is stronger
  than Paraformer English text, but the current speaker-labeled pipeline output
  and memory cost fail the local-gateway cost-benefit bar.

- [ ] **Step 3: Review the public docs diff**

  Run:

  ```bash
  git diff -- README.md MODELS.md
  ```

  Expected: only documentation wording changes; no runtime code or config
  changes.

### Task 2: Decision and Spec Alignment

**Files:**
- Modify: `docs/ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md`
- Modify: `docs/SPEC-013-Production-Decoupled-Diarization-Pipeline.md`
- Modify if useful: `docs/superpowers/2026-05-19-qwen3-sortformer-longform-defects.md`

- [ ] **Step 1: Reframe ADR-002**

  Replace production-reachability wording for `qwen3-sortformer` with
  experimental reachability. Make deletion an acceptable future outcome and
  preserve the no-more-local-diarization-recovery boundary.

- [ ] **Step 2: Add the real 1:1 probe reassessment to SPEC-013**

  Record the 2026-05-20 English manager 1:1 evidence:

  ```text
  qwen3-sortformer: stronger top-level English text, but segments coverage and
  segmentation trustworthiness failed the target use case; peak resource cost
  was too high for the local gateway value delivered.
  paraformer: structurally coherent speaker segments on the sample, but worse
  English transcript quality.
  ```

  Mark the Sortformer route as experimental rather than an active production
  target unless upstream capabilities materially change the tradeoff.

- [ ] **Step 3: Link earlier defect evidence to the newer reassessment if needed**

  If the 2026-05-19 long-form defect note still reads as a pending enablement
  gate, add a short follow-up note pointing to the later real-meeting
  reassessment.

- [ ] **Step 4: Review the decision-doc diff**

  Run:

  ```bash
  git diff -- docs/ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md docs/SPEC-013-Production-Decoupled-Diarization-Pipeline.md docs/superpowers/2026-05-19-qwen3-sortformer-longform-defects.md
  ```

  Expected: documentation explains the downgrade without adding a new technical
  repair plan.

### Task 3: Research Notes Alignment

**Files:**
- Modify: `/Users/leipeng/Documents/research-notes/03-project/2026-03_local-asr-service-MOC.md`
- Create: `/Users/leipeng/Documents/research-notes/01-inbox/2026-05-21_local-asr-service_qwen3-sortformer-real-meeting-reassessment.md`

- [ ] **Step 1: Update the project MOC**

  Replace stale language that treats `qwen3-sortformer` as the next enablement
  target with the new decision: it is an experimental reachable path and
  deletion candidate after the real 1:1 probe.

- [ ] **Step 2: Create the reassessment research note**

  Use the Research Notes architecture-analysis style. Capture:

  - why this 1:1 sample is more representative than a promising probe;
  - `qwen3-sortformer` transcript-quality benefit versus segment-quality and
    resource-cost failure;
  - `paraformer` structural consistency versus English-quality failure;
  - the project lesson that a 64 GB Mac being able to run a path does not make
    it a good local gateway feature;
  - the future preference for replacing the route with a stronger upstream or
    open-source local English diarized ASR solution.

- [ ] **Step 3: Review Research Notes consistency**

  Search the touched notes for stale `production` or `enablement` language that
  contradicts the new decision before finalizing.

### Task 4: Documentation Verification

**Files:**
- Verify all touched markdown files.

- [ ] **Step 1: Confirm scope stayed documentation-only**

  Run:

  ```bash
  git diff --name-only
  ```

  Expected: markdown documentation files only.

- [ ] **Step 2: Search for stale public positioning**

  Run:

  ```bash
  rg -n "qwen3-sortformer|Production Reachability|discovery-only|production" README.md MODELS.md docs/ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md docs/SPEC-013-Production-Decoupled-Diarization-Pipeline.md docs/superpowers/2026-05-19-qwen3-sortformer-longform-defects.md
  ```

  Expected: any remaining `production` wording is historical context or clearly
  describes a rejected/paused production direction.

- [ ] **Step 3: Summarize touched docs and future removal path**

  Final response should state that no runtime code changed and that future code
  review/removal can start from the now-documented deletion-candidate posture.
