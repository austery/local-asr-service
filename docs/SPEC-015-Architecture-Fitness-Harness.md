---
specId: SPEC-015
title: Architecture Fitness Harness
status: ✅ 已完成 (Completed)
priority: P3 - Quality
creationDate: 2026-05-19
lastUpdateDate: 2026-05-19
owner: User (AI-Assisted)
relatedSpecs:
  - ADR-002
tags:
  - architecture-boundary
  - fitness-function
  - quality-gate
---

# SPEC-015: Architecture Fitness Harness

## 1. Goal

Establish an automated, lightweight, and unified architecture and quality harness to enforce the project boundaries defined in [ADR-002](file:///Users/leipeng/Documents/Projects/local-asr-service/docs/ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md) and prevent boundary creep.

## 2. Background

In [ADR-002](file:///Users/leipeng/Documents/Projects/local-asr-service/docs/ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md), we defined the `local-asr-service` as a lightweight OpenAI-compatible local speech gateway, not a general speech model platform. However, without automated guardrails, the codebase remains susceptible to incremental creep:
- Import dependency leaks (e.g., inner layers importing from outer layers).
- Cognitive complexity accumulation (e.g., long helper functions bypass review).
- Registry inconsistency (e.g., experimental models marked requestable or missing from `MODELS.md`).

To maintain consistency and simplify the maintenance burden, we adopt the exact same tool stack utilized in other Python projects (`BookWeaver`, `distil-nexus`): **Tach** for dependency boundaries, and **Ruff** for code complexity. Custom rules are implemented directly in the test suite using standard **pytest** capabilities.

## 3. Design Decision

**Chosen approach**: Enforce dependency flows and code complexity checks natively via `Tach` configurations and `Ruff` linters, paired with a custom `pytest` test suite to check semantic constraints.

### Comparison of boundary tools for Python

| Tool | Pros | Cons | Decision |
|------|------|------|----------|
| **import-linter** | Declarative configuration. | Not used in other projects. | ❌ Rejected |
| **pytest-archon** | Code-based tests similar to ArchUnit. | Different style, slightly slower. | ❌ Rejected |
| **Tach** | Written in Rust (extremely fast), declarative config (`tach.toml`), supports circular dependency checking. Already used in `BookWeaver` and `distil-nexus`. | Requires extra dependency. | ✅ Chosen (consistency & simplicity) |

---

## 4. Implementation Details

### 4.1. Module & Dependency Rules (`Tach`)
Create `tach.toml` at the project root to enforce layer boundaries:

- **`src.adapters`**: Core utility layer. Depends on `src.core` (for type definitions like `AlignedWord`, `SpeakerTurn`) and `src.config`.
- **`src.core`**: Inference engines and registry. Depends on `src.adapters` (helpers) and `src.config`.
- **`src.workers`**: Subprocess implementation. Depends on `src.core` and `src.config`.
- **`src.services`**: Orchestration and serialization. Depends on `src.core`, `src.adapters`, `src.workers`, and `src.config`.
- **`src.api`**: Interface. Depends on `src.services`, `src.core`, and `src.config`.
- **`src.config`**: Utility. Zero dependencies.

Circular dependencies at the file-import level are strictly forbidden (`forbid_circular_dependencies = true`).

### 4.2. Cognitive Complexity & Method Length (`Ruff`)
We configure `Ruff` in `pyproject.toml` to fail on excessively complex methods:
- ** McCabe Cyclomatic Complexity (`C901`)**: Set `max-complexity = 10` (industry standard).
- ** Pylint Refactoring Rules (`PLR`)**:
  - `PLR0915` (Too many statements): Set `max-statements = 50`.
  - `PLR0912` (Too many branches): Set `max-branches = 12`.
  - `PLR0913` (Too many arguments): Set `max-args = 5`.

### 4.3. Project-Specific Semantic Checks (`pytest`)
A dedicated unit test file [test_architecture_fitness.py](file:///Users/leipeng/Documents/Projects/local-asr-service/tests/unit/test_architecture_fitness.py) is introduced to statically verify the following draft requirements:
1. **`MODELS.md` Sync**: Automatically scans the model and pipeline registries, verifying that all registered aliases are documented.
2. **Explicit `requestable`**: Checks that all pipeline profiles explicitly declare `requestable` in their definition, rather than relying on defaults.
3. **Module Gating**: Asserts that only approved engine adapters and diarization reconciliation modules exist, blocking undocumented extensions.
4. **Service Scope Check**: Scans `TranscriptionService` methods to ensure no job domains outside `transcribe`, `align`, and `diarize` are added.
5. **Config Safety**: Verifies that the default model in `src/config.py` is requestable and is not marked as experimental.

---

## 5. Implementation Phases

### Phase 1: Dependency Setup & Linter Configuration
- [x] Add `tach>=0.34.0` to `dependency-groups.dev` in [pyproject.toml](file:///Users/leipeng/Documents/Projects/local-asr-service/pyproject.toml).
- [x] Enable `C901` and `PLR` rules in Ruff's `select` list in `pyproject.toml`.
- [x] Run `uv run ruff check .` and fix or suppress any existing violations.
- [x] Run `uv sync`.

### Phase 2: Tach Boundary Specification
- [x] Create [tach.toml](file:///Users/leipeng/Documents/Projects/local-asr-service/tach.toml) configuration.
- [x] Execute `uv run tach check` to verify existing imports and correct any architectural leaks.

### Phase 3: Custom Fitness Tests
- [x] Create [test_architecture_fitness.py](file:///Users/leipeng/Documents/Projects/local-asr-service/tests/unit/test_architecture_fitness.py).
- [x] Verify test suite execution: `uv run pytest tests/unit/test_architecture_fitness.py`.

---

## 6. Acceptance Criteria

- [x] `uv run tach check` passes successfully without dependency errors.
- [x] `uv run ruff check .` passes successfully, enforcing a maximum cyclomatic complexity of 10 and max statements of 50.
- [x] `uv run pytest tests/unit/test_architecture_fitness.py` passes successfully, verifying document sync and registry constraints.

---

## 7. Status History

| Date | Status | Note |
|------|--------|------|
| 2026-05-19 | 📝 草案 (Draft) | Initial draft |
| 2026-07-06 | ✅ 已完成 (Completed) | Fully implemented, verified via Ruff, Tach, and Pytest fitness checks |

---

## 8. Related

- **Code**: [pyproject.toml](file:///Users/leipeng/Documents/Projects/local-asr-service/pyproject.toml)
- **ADRs**: [ADR-002](file:///Users/leipeng/Documents/Projects/local-asr-service/docs/ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md)
