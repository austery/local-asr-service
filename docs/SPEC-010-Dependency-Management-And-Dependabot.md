---
specId: SPEC-010
title: Dependency Management and Dependabot Configuration Strategy
status: 🟢 已实施，Phase 3 监控未完成 (Implemented; Phase 3 monitoring open)
priority: P1 - Core Infrastructure
creationDate: 2026-03-16
lastUpdateDate: 2026-11-22
owner: Copilot (AI-Assisted)
relatedSpecs:
  - SPEC-001
  - SPEC-002
  - SPEC-009
tags:
  - dependency-management
  - version-control
  - python-ecosystem
  - apple-silicon
  - mlx
  - pytorch
  - funasr
---

# SPEC-010: Dependency Management and Dependabot Configuration Strategy

## 1. Goal

Establish a centralized, version-constraint strategy for Python dependencies to prevent breaking changes in dual-engine inference (PyTorch/FunASR + MLX Audio) while maintaining security patch flow and architectural consistency with puresubs (TypeScript/Angular) management patterns.

## 2. Background

### Current State (historical, 2026-03-16; superseded by the files named in §10)

- **No Dependabot configuration**: `.github/dependabot.yml` missing
- **No version constraints**: `pyproject.toml` uses loose version ranges (e.g., `torch>=2.9.1`, no upper bound)
- **Active architecture evolution**: Project continuously refactors (subprocess isolation, idle model offload, dynamic model switching)
- **Dual-engine design**: PyTorch (FunASR) and MLX Audio coexist — version incompatibilities could silently break either engine
- **Reference project**: puresubs (TypeScript/Angular) has mature Dependabot strategy with version constraints and grouping

### Pain Points

1. **Silent breaking changes**: No protection against major version bumps (e.g., mlx-audio 0.4.0, PyTorch 3.0, FunASR 2.0)
2. **Prerelease risk**: mlx-audio 0.3.1 is prerelease; Dependabot could auto-upgrade to unstable 0.4.0
3. **Monolithic PR flood**: Without grouping, each dependency update creates separate PR
4. **Manual dependency audits**: No systematic check for PyTorch ↔ MLX ↔ FunASR compatibility
5. **Inconsistent strategy**: puresubs has mature patterns; local-asr-service has none

### Constraints

- **Python 3.11–3.12**: `requires-python = ">=3.11,<3.13"`
- **Apple Silicon native**: Must support MLX (MPS Metal backend)
- **macOS+Linux**: PyTorch also required for fallback/Linux deployments
- **Subprocess architecture (SPEC-009)**: Model worker isolation increases criticality of stable dependencies

## 3. Design Decision

### Chosen Approach

**Create `.github/dependabot.yml` with version constraints and grouping strategy**, mirroring puresubs patterns but adapted for Python ecosystem:

1. **Version constraint rules** (ignore unstable major versions)
   - `mlx-audio >=0.4.0` → defer until stable and tested
   - `torch <3.0` → prevent major version breaking changes
   - `torchaudio <3.0` → lock to torch version
   - `funasr <2.0` → defer major version until tested

2. **Dependency grouping** (reduce PR noise)
   - `security-patches`: All production patches (highest priority)
   - `production-updates`: Minor updates to production deps
   - `development-updates`: Minor/patch updates to dev deps

3. **Update schedule** (weekly, aligned with puresubs)
   - Monday 09:00 UTC (17:00 Beijing)
   - Limit to 5 concurrent PRs

4. **Commit convention** (Conventional Commits)
   - Prefix: `chore`
   - Include scope: `chore(deps): upgrade ...`

### Rationale

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| No Dependabot + manual audits | Full control, no automation | High maintenance, easy to miss security patches | ❌ |
| Dependabot, no version constraints | Catch all updates automatically | Breaking changes silently in CI, high noise | ❌ |
| **Dependabot + version constraints + grouping** | Security patches flow, breaking changes blocked, organized PRs, aligns with puresubs | Slight delay for major versions | ✅ **Chosen** |
| All major versions blocked (very strict) | Zero breaking changes | Stale dependencies, harder to upgrade later | ❌ |

## 4. Implementation Phases

### Phase 1: Create `.github/dependabot.yml` — Target: 2026-03-16 (This Week)

- [x] Create `.github/` directory structure
- [x] Write `.github/dependabot.yml` with version constraints and grouping rules
- [ ] Document version constraint rationale in CLAUDE.md (Known Issues section)
- [x] Add to git and push to remote (auto-enable on GitHub)
- [x] Verify Dependabot is active in repo settings

**Acceptance**: Dependabot dashboard shows "Active" and pulls latest dependency list

### Phase 2: Update `pyproject.toml` with Version Boundaries — Target: 2026-03-16 (This Week)

- [x] Add `torch<3.0` constraint to pyproject.toml (line 8)
- [x] Add `torchaudio<3.0` constraint (line 9)
- [ ] Document rationale in comment: `# <3.0: Apple Silicon Metal compatibility, avoid PyTorch 3.0 breaking changes`
- [ ] Run `uv sync` and validate lock file
- [ ] Commit with message: `chore: add version upper bounds for torch/torchaudio for stability`

**Acceptance**: `uv sync` succeeds, `uv.lock` reflects new constraints

### Phase 3: Backlog — FunASR and MLX Audio Monitoring (Ongoing)

- [ ] Add mlx-audio 0.4.0 release monitor (GitHub release watcher, or quarterly manual check)
- [ ] Add FunASR 2.0 release monitor (quarterly check)
- [ ] When mlx-audio 0.4.0 stable lands:
  - Run full E2E test suite with prerelease
  - If stable: remove `mlx-audio <0.4` ignore rule
  - If unstable: update Dependabot comment with rationale
- [ ] Document in CLAUDE.md > Known Issues updates

**Acceptance**: Quarterly audit log shows monitoring status

## 5. Version Compatibility Status (Baseline)

Historical baseline from 2026-03-16. The versions below are not current; `uv.lock` is the source of truth. The constraint table after it was updated on 2026-11-22.

### ✅ Verified Compatible

| Pair | Current Versions | Status | Tested? |
|------|------------------|--------|---------|
| PyTorch ↔ torchaudio | 2.9.1 ↔ 2.9.1 | ✅ Paired | Yes |
| PyTorch ↔ MLX | 2.9.1 ↔ 0.30.4 | ✅ Independent | N/A |
| MLX Framework ↔ MLX Audio | 0.30.4 ↔ 0.3.1 | ✅ Dependency chain | Yes |
| FunASR ↔ PyTorch | 1.2.7 ↔ 2.9.1 | ✅ Indirect torch-complex | Yes |
| Transformers (shared) | 4.46.x | ✅ Coordinated by uv.lock | Yes |
| Python version | 3.11–3.12 | ✅ All supported | Yes |

### ⚠️ Constraints to Monitor

| Constraint | Rationale | Review Period |
|-----------|-----------|---|
| `mlx-audio <0.4` | Retired. Never added to `.github/dependabot.yml`, and mlx-audio is now 0.6.6. Runtime upgrades follow §5a | Not applicable |
| `torch <3.0` | PyTorch 3.0 planned for late 2026; unknown Metal/MPS compatibility | Quarterly or on 3.0 alpha |
| `torchaudio <3.0` | Must track torch version | With torch monitoring |
| `funasr` (all versions) | Dependabot version updates are ignored (major, minor, patch). Upgrades are manual, after §5a and a real-audio A/B. Reason: the 1.4.19 upgrade turned silent missing timestamps into a 500 on a 52-minute request | When a release contains FunASR#3935 (issue #88) |

## 5a. When a Runtime Upgrade Is Worth Doing

Scope: the inference runtimes in `uv.lock` (`funasr`, `torch`, `torchaudio`,
`mlx`, `mlx-audio`, `mlx-metal`). Owner decision, 2026-11-13. A new release
alone is not a reason to upgrade. Start an upgrade only when every condition
below holds:

1. **Released.** The fix or feature is in a published PyPI release. A merged
   upstream PR is not enough.
2. **Reachable.** The changed upstream code is on a path a production alias
   runs (`paraformer`, `qwen3-asr`, `sensevoice-small`, `moss-transcribe-diarize`).
   Trace the call down to the class the cached model's config selects. A
   changelog title is not evidence: the 2026-11-13 review found that the
   "Paraformer" CIF fix (#3854) does not reach SEACO Paraformer, which uses
   `CifPredictorV3`.
3. **Measurable benefit.** The change fixes a failure we observed or can
   reproduce, or improves an outcome we can measure on the approved clips. A
   cosmetic text change does not count as benefit by itself.
4. **Known regressions accepted.** Every behavior change inherited from the
   intermediate releases is either judged acceptable by the owner or outweighed
   by the benefit. Example: the FunASR 1.4.23 VAD end-silence change split one
   speaker into two on the 300-second clip, and every later FunASR release
   carries it ([radius map](agents/impact-radius.md), Dependencies).
5. **A/B cost accepted.** The owner agrees to run the real-model A/B on the
   approved clips that the radius map requires for a runtime change
   ("Structure = touched"). It uses cached weights only, and it never runs
   beside the live service.

If any condition fails, record the decision and its re-evaluation trigger in
the status history below, and wait. Prefer one upgrade that collects several
released fixes over a series of small upgrades: each upgrade costs one A/B.
Security fixes follow the Dependabot path and are assessed separately.

## 6. Acceptance Criteria

- [ ] `.github/dependabot.yml` exists and is syntactically valid
- [ ] GitHub Dependabot dashboard shows "Active" status
- [ ] `pyproject.toml` has `torch<3.0` and `torchaudio<3.0` constraints
- [ ] `uv sync` succeeds with new constraints
- [ ] First Dependabot PR is created (within 7 days of config creation)
- [ ] PR is correctly grouped (security, production, or dev group)
- [ ] PR commit message follows Conventional Commits format: `chore(deps): ...`
- [ ] CLAUDE.md or project README documents version constraints and monitoring strategy
- [ ] E2E test suite passes (validates PyTorch + MLX coexistence with new constraints)

## 7. Configuration Reference

### `.github/dependabot.yml`

The live file is `.github/dependabot.yml`. This spec no longer copies it, because
the copy drifted. What the file does, as of 2026-11-22:

- `pip` ecosystem, weekly on Monday 09:00 UTC, at most 5 open PRs, grouped into
  production patches, production minors and development updates.
- Ignored: `torch` and `torchaudio` major versions; every `funasr` version
  update (major, minor, patch).
- `github-actions` ecosystem, monthly, one group.

### `pyproject.toml` (Version Constraint Updates)

Add to line 8 and 9 (existing torch/torchaudio entries):

```toml
dependencies = [
    "fastapi>=0.121.3",
    "funasr[llm]>=1.2.7",
    "huggingface-hub>=0.26.0",
    "mlx-audio>=0.3.0",
    "python-dotenv>=1.0.0",
    "python-multipart>=0.0.20",
    "torch>=2.9.1,<3.0",          # ← NEW: <3.0 constraint
    "torchaudio>=2.9.1,<3.0",     # ← NEW: <3.0 constraint (must track torch)
    "uvicorn>=0.38.0",
]
```

## 8. Related Architecture Decisions

- **SPEC-009 (Idle Model Offload)**: Subprocess isolation increases dependency stability criticality
- **SPEC-008 (Dynamic Model Switching)**: Model registry depends on stable transformers/pytorch versions
- **ADR-001**: (if exists) Engine abstraction decouples interface from implementation version changes
- **puresubs Dependabot Config** (reference): TypeScript/Angular strategy mirrors this pattern

## 9. Status History

| Date | Status | Note |
|------|--------|------|
| 2026-03-16 | 🟡 待实施 (Ready for Implementation) | Initial analysis complete, awaiting implementation |
| 2026-11-13 | Runtime upgrades deferred | Reviewed against §5a. FunASR stays at 1.4.19 (latest 1.4.26): no released fix reaches the production path with a measurable benefit, and 1.4.23+ carries the speaker-split change. Re-evaluate when upstream releases #3848 (empty-ASR result row) or #3862 (pure-silence VAD segments). mlx-audio stays at 0.6.1 (latest 0.6.5): Qwen3-ASR #1155 (per-chunk token budget) and the Whisper fixes #1164/#1165 merged after 0.6.5. Re-evaluate when a release contains #1155. |
| 2026-11-17 | Owner-authorized branch trial | mlx-audio 0.6.6 contains #1155. The owner waived the full A/B matrix for this task, requested tests using existing local audio, and authorized stopping the old service and serving real traffic from the branch. requests 2.33.0 is the required companion update. This is a task-specific exception, not a change to section 5a. FunASR stays at 1.4.19. See [trial scope](plans/2026-11-17-mlx-audio-066-upgrade-assessment.md). Merge remains unapproved. |
| 2026-11-22 | Status corrected; FunASR held | `.github/dependabot.yml` exists and Dependabot has opened PRs, so the 2026-03-16 "not implemented" status was stale. Section 7 now points at the live file. `funasr` minor and patch version updates are now ignored, in addition to major. FunASR#3935 (the upstream fix for our #3934) merged on 2026-11-22 but is unreleased; PyPI 1.4.28 does not contain it. Re-evaluate when a release contains it (issue #88). |

## 10. Related

- **Code**: `pyproject.toml`, `.github/dependabot.yml`
- **Architecture**: SPEC-009 (Idle Model Offload), SPEC-008 (Dynamic Model Switching)
- **Reference**: puresubs `.github/dependabot.yml` (TypeScript/Angular pattern)
- **Knowledge Base**: CLAUDE.md > Known Issues & Architecture Decisions
