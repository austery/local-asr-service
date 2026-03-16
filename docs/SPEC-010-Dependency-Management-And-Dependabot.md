---
specId: SPEC-010
title: Dependency Management and Dependabot Configuration Strategy
status: 🟡 待实施 (Ready for Implementation)
priority: P1 - Core Infrastructure
creationDate: 2026-03-16
lastUpdateDate: 2026-03-16
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

### Current State

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

- [ ] Create `.github/` directory structure
- [ ] Write `.github/dependabot.yml` with version constraints and grouping rules
- [ ] Document version constraint rationale in CLAUDE.md (Known Issues section)
- [ ] Add to git and push to remote (auto-enable on GitHub)
- [ ] Verify Dependabot is active in repo settings

**Acceptance**: Dependabot dashboard shows "Active" and pulls latest dependency list

### Phase 2: Update `pyproject.toml` with Version Boundaries — Target: 2026-03-16 (This Week)

- [ ] Add `torch<3.0` constraint to pyproject.toml (line 8)
- [ ] Add `torchaudio<3.0` constraint (line 9)
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
| `mlx-audio <0.4` | 0.3.1 is prerelease; 0.4.0 may have breaking API changes | When 0.4.0 stable released |
| `torch <3.0` | PyTorch 3.0 planned for late 2026; unknown Metal/MPS compatibility | Quarterly or on 3.0 alpha |
| `torchaudio <3.0` | Must track torch version | With torch monitoring |
| `funasr <2.0` | No monitoring yet; upgrade deferred until major version released | Quarterly or on 2.0 alpha |

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

### `.github/dependabot.yml` (Complete)

```yaml
version: 2

updates:
  # Python dependencies (pip/uv)
  - package-ecosystem: 'pip'
    directory: '/'
    schedule:
      interval: 'weekly'
      day: 'monday'
      time: '09:00'  # 17:00 Beijing time

    # Version constraints: prevent breaking changes in dual-engine design
    ignore:
      # MLX Audio: defer 0.4+ until stable and thoroughly tested
      - dependency-name: 'mlx-audio'
        versions: ['>=0.4.0']

      # PyTorch: block major version 3.0+ (unknown Apple Silicon/Metal compatibility)
      - dependency-name: 'torch'
        update-types: ['version-update:semver-major']

      # torchaudio: must track torch version, block 3.0+
      - dependency-name: 'torchaudio'
        update-types: ['version-update:semver-major']

      # FunASR: defer major version 2.0+ until tested with current PyTorch
      - dependency-name: 'funasr'
        update-types: ['version-update:semver-major']

    # Grouping strategy: reduce PR noise, organize by priority
    groups:
      # Group 1: Production security patches (highest priority)
      security-patches:
        patterns:
          - '*'
        update-types:
          - 'patch'
        dependency-type: 'production'

      # Group 2: Production minor updates
      production-updates:
        patterns:
          - '*'
        update-types:
          - 'minor'
        dependency-type: 'production'

      # Group 3: Development updates (lower priority)
      development-updates:
        patterns:
          - '*'
        update-types:
          - 'minor'
          - 'patch'
        dependency-type: 'development'

    # Other settings
    open-pull-requests-limit: 5
    labels:
      - 'dependencies'
      - 'automated'

    # Conventional Commits format
    commit-message:
      prefix: 'chore'
      prefix-development: 'chore'
      include: 'scope'

  # GitHub Actions updates (separate ecosystem)
  - package-ecosystem: 'github-actions'
    directory: '/'
    schedule:
      interval: 'monthly'
    groups:
      github-actions:
        patterns:
          - '*'
    labels:
      - 'dependencies'
      - 'github-actions'
```

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

## 10. Related

- **Code**: `pyproject.toml`, `.github/dependabot.yml` (to be created)
- **Architecture**: SPEC-009 (Idle Model Offload), SPEC-008 (Dynamic Model Switching)
- **Reference**: puresubs `.github/dependabot.yml` (TypeScript/Angular pattern)
- **Knowledge Base**: CLAUDE.md > Known Issues & Architecture Decisions
