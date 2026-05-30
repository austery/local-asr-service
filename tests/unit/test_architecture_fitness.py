import ast
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

from src.core.model_registry import list_all as list_all_models
from src.core.pipeline_registry import list_all_profiles

ALLOWED_JOB_KINDS = ("transcribe", "align", "diarize")

# Layer 2 Hard Gate: maximum block nesting depth allowed in any src/ function.
# Set conservatively at 5; tighten once orchestrator refactors land.
_MAX_NESTING_DEPTH = 5

# Files whose functions legitimately exceed _MAX_NESTING_DEPTH.
# DO NOT add entries without a design rationale. Depth value is the approved ceiling.
_NESTING_DEPTH_ALLOWLIST: dict[str, int] = {
    # SPEC-009 subprocess IPC worker loop: outer-try → inner-try → while → try(dequeue)
    # → if(job_kind) → if(alias-validation) is structural; extracting helpers would obscure
    # the IPC contract without reducing real complexity.
    "src/workers/model_worker.py": 6,
}

# Governance: Ruff rules considered "complexity suppressions".
_COMPLEXITY_RULES: frozenset[str] = frozenset({"C901", "PLR0911", "PLR0912", "PLR0915"})

# Files approved to suppress complexity rules — do NOT add entries without an ADR or design rationale.
_APPROVED_COMPLEXITY_SUPPRESSED_FILES: frozenset[str] = frozenset({
    "tests/**/*.py",              # test code: complex mocks and setup are expected
    "src/api/routes.py",          # SPEC-014: multi-engine dispatch, approved in ADR-002
    "src/services/transcription.py",  # SPEC-009: subprocess IPC orchestrator complexity
    "src/workers/model_worker.py",    # subprocess worker loop: inherent IPC complexity
    "benchmarks/**/*.py",         # benchmarks: measurement scaffolding
    "examples/**/*.py",           # examples: demo scripts
})

_BLOCK_STMT_TYPES: tuple[type[ast.stmt], ...] = (
    ast.If,
    ast.For,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFor,
)


def _workspace_root() -> Path:
    return Path(__file__).parent.parent.parent


def _parse_python_file(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal_values(annotation: ast.expr | None) -> set[str]:
    if not isinstance(annotation, ast.Subscript):
        return set()
    if not isinstance(annotation.value, ast.Name) or annotation.value.id != "Literal":
        return set()

    slice_node = annotation.slice
    values = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]
    return {node.value for node in values if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def _getenv_default_string(node: ast.AST, variable_name: str) -> str:
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == variable_name for target in child.targets):
            continue
        if (
            isinstance(child.value, ast.Call)
            and isinstance(child.value.func, ast.Attribute)
            and child.value.func.attr == "getenv"
            and len(child.value.args) >= 2
            and isinstance(child.value.args[1], ast.Constant)
            and isinstance(child.value.args[1].value, str)
        ):
            return child.value.args[1].value
    raise AssertionError(f"Could not find string default for {variable_name} in config.py")


def _child_stmt_bodies(stmt: ast.stmt) -> list[list[ast.stmt]]:
    """Return all direct child statement lists of a compound statement.

    Handles ast.ExceptHandler (whose .body is not exposed as a plain list field).
    """
    result: list[list[ast.stmt]] = []
    for field, value in ast.iter_fields(stmt):
        if field == "handlers":
            for handler in value:
                if isinstance(handler, ast.ExceptHandler):
                    result.append(handler.body)
        elif isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            result.append(value)  # type: ignore[arg-type]
    return result


def _max_block_depth(body: list[ast.stmt], depth: int = 0) -> int:
    """Return the maximum block nesting depth in a statement list.

    elif chains are treated as flat (same depth as the parent if) so that
    long if/elif/elif sequences do not inflate the nesting score.  A real
    else: block is treated as being at the same nesting level as the if body
    (child_depth), because it IS a separate nested block.

    The distinction is made structurally: if orelse contains exactly one ast.If
    node it is an elif chain; anything else is a genuine else block.

    Nested function/class definitions are not crossed — they have their own
    nesting budget.
    """
    max_d = depth
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not isinstance(stmt, _BLOCK_STMT_TYPES):
            continue
        child_depth = depth + 1
        max_d = max(max_d, child_depth)
        if isinstance(stmt, ast.If):
            max_d = max(max_d, _max_block_depth(stmt.body, child_depth))
            # Distinguish elif from else: if using source-position data.
            # An `elif` keyword is always at the same indentation (col_offset) as
            # its parent `if`, while the `if` inside a real `else:` block is indented
            # further.  Both produce orelse=[ast.If(...)], so the node type alone
            # is not sufficient.
            is_elif = (
                len(stmt.orelse) == 1
                and isinstance(stmt.orelse[0], ast.If)
                and stmt.orelse[0].col_offset == stmt.col_offset
            )
            if is_elif:
                # elif chain: keep flat — the elif keyword is at the same indentation
                max_d = max(max_d, _max_block_depth(stmt.orelse, depth))
            else:
                # genuine else block (including else: if): counts as child_depth
                max_d = max(max_d, _max_block_depth(stmt.orelse, child_depth))
        else:
            for child_body in _child_stmt_bodies(stmt):
                max_d = max(max_d, _max_block_depth(child_body, child_depth))
    return max_d


def test_models_md_documents_all_aliases() -> None:
    """Verify that all model aliases and pipeline profiles in the registry are documented in MODELS.md."""
    workspace_root = _workspace_root()
    models_md_path = workspace_root / "MODELS.md"
    assert models_md_path.exists(), "MODELS.md does not exist at project root"

    content = models_md_path.read_text(encoding="utf-8")

    # Find all inline code blocks like `alias` in MODELS.md
    documented_aliases = set(re.findall(r"`([^`\s]+)`", content))

    # All model registry aliases must be documented
    model_aliases = [spec.alias for spec in list_all_models()]
    assert len(model_aliases) > 0, "Model registry is empty"

    for alias in model_aliases:
        assert alias in documented_aliases, (
            f"Model alias '{alias}' registered in model_registry.py is not documented in MODELS.md. "
            f"Please update MODELS.md."
        )

    # All pipeline profile aliases must be documented
    profile_aliases = [profile.alias for profile in list_all_profiles()]
    for alias in profile_aliases:
        assert alias in documented_aliases, (
            f"Pipeline profile alias '{alias}' registered in pipeline_registry.py is not documented in MODELS.md. "
            f"Please update MODELS.md."
        )


def test_pipeline_profiles_declare_requestable_explicitly() -> None:
    """Ensure all PipelineProfile instances in pipeline_registry.py explicitly declare the 'requestable' field."""
    workspace_root = _workspace_root()
    registry_path = workspace_root / "src" / "core" / "pipeline_registry.py"
    assert registry_path.exists(), "pipeline_registry.py does not exist"

    tree = _parse_python_file(registry_path)

    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "PipelineProfile"
        ):
            calls.append(node)

    assert len(calls) > 0, "No PipelineProfile instantiations found in pipeline_registry.py"

    for call in calls:
        has_requestable = False
        for keyword in call.keywords:
            if keyword.arg == "requestable":
                has_requestable = True
                break
        assert has_requestable, (
            "PipelineProfile instantiations in pipeline_registry.py must explicitly define the 'requestable' argument "
            "to prevent unintentional exposure of experimental pipelines (e.g., requestable=True/False)."
        )


def test_engine_adapters_and_diarization_are_gated() -> None:
    """Gate core engine adapters and utility adapters.

    Any changes to this list must be accompanied by updates to fitness tests and architecture documents.
    """
    workspace_root = _workspace_root()

    # Enforce src/core files
    core_dir = workspace_root / "src" / "core"
    assert core_dir.exists(), "src/core directory does not exist"

    allowed_core_files = {
        "__init__.py",
        "base_engine.py",
        "alignment_port.py",
        "diarization_port.py",
        "alignment_registry.py",
        "diarization_registry.py",
        "factory.py",
        "funasr_engine.py",
        "mlx_engine.py",
        "mlx_qwen_forced_aligner.py",
        "mlx_sortformer_diarizer.py",
        "model_registry.py",
        "pipeline_registry.py",
    }

    current_core_files = {
        f.name for f in core_dir.iterdir() if f.is_file() and not f.name.startswith(".")
    }

    # Assert no undocumented engines/files were added under core
    unexpected_core = current_core_files - allowed_core_files
    assert not unexpected_core, (
        f"Unexpected files found in src/core: {unexpected_core}. "
        f"To add a new ASR engine or adapter, you must document it in ADR-002, "
        f"satisfy SPEC-014 rules, and update allowed_core_files list in this test."
    )

    # Enforce src/adapters files
    adapters_dir = workspace_root / "src" / "adapters"
    assert adapters_dir.exists(), "src/adapters directory does not exist"

    allowed_adapters_files = {
        "__init__.py",
        "audio_chunking.py",
        "pipeline_chunking.py",
        "segment_alignment.py",
        "text.py",
    }

    current_adapters_files = {
        f.name for f in adapters_dir.iterdir() if f.is_file() and not f.name.startswith(".")
    }

    # Assert no undocumented helpers were added under adapters
    unexpected_adapters = current_adapters_files - allowed_adapters_files
    assert not unexpected_adapters, (
        f"Unexpected files found in src/adapters: {unexpected_adapters}. "
        f"To add a new processing helper, you must register it in this test."
    )


def _check_function_def_job_kind(
    node: ast.AsyncFunctionDef | ast.FunctionDef,
    allowed_kinds: tuple[str, ...],
) -> None:
    if node.name not in ("_submit_worker_job", "_enqueue_worker_job"):
        return
    job_kind_idx = -1
    job_kind_arg: ast.arg | None = None
    for idx, arg in enumerate(node.args.args):
        if arg.arg == "job_kind":
            job_kind_idx = idx
            job_kind_arg = arg
            break
    if job_kind_idx == -1:
        return

    literal_values = _literal_values(job_kind_arg.annotation if job_kind_arg else None)
    assert literal_values == set(allowed_kinds), (
        f"{node.name} job_kind annotation must be exactly Literal{allowed_kinds}; "
        f"got {literal_values}."
    )

    defaults_start_idx = len(node.args.args) - len(node.args.defaults)
    if job_kind_idx >= defaults_start_idx:
        default_node = node.args.defaults[job_kind_idx - defaults_start_idx]
        if isinstance(default_node, ast.Constant):
            assert default_node.value in allowed_kinds, (
                f"Default value '{default_node.value}' for job_kind in {node.name} "
                f"must be one of {allowed_kinds}."
            )


def _check_call_job_kind(node: ast.Call, allowed_kinds: tuple[str, ...]) -> None:
    func_name = ""
    if isinstance(node.func, ast.Attribute):
        func_name = node.func.attr
    elif isinstance(node.func, ast.Name):
        func_name = node.func.id

    if func_name not in ("_submit_worker_job", "_enqueue_worker_job"):
        return

    job_kind_node = None
    for kw in node.keywords:
        if kw.arg == "job_kind":
            job_kind_node = kw.value
            break

    if job_kind_node is None:
        idx = 5 if func_name == "_submit_worker_job" else 6
        if len(node.args) > idx:
            job_kind_node = node.args[idx]

    if job_kind_node is not None:
        if isinstance(job_kind_node, ast.Constant):
            val = job_kind_node.value
            assert val in allowed_kinds, (
                f"Disallowed job_kind '{val}' passed to {func_name} at line {node.lineno}. "
                f"Must be one of {allowed_kinds}."
            )
        elif isinstance(job_kind_node, ast.Name) and job_kind_node.id == "job_kind":
            # Legitimate parameter forwarding from _submit_worker_job to _enqueue_worker_job
            pass
        else:
            raise AssertionError(
                f"Non-constant/dynamic expression passed for job_kind to {func_name} "
                f"at line {node.lineno}. The job_kind must be a compile-time string literal "
                f"within {allowed_kinds} or direct parameter forwarding of 'job_kind'."
            )


def test_transcription_service_job_domains() -> None:
    """Verify that the job kind domains in TranscriptionService are strictly limited to transcribe/align/diarize."""
    workspace_root = _workspace_root()
    service_path = workspace_root / "src" / "services" / "transcription.py"
    assert service_path.exists(), "transcription.py does not exist"

    tree = _parse_python_file(service_path)

    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            _check_function_def_job_kind(node, ALLOWED_JOB_KINDS)
        elif isinstance(node, ast.Call):
            _check_call_job_kind(node, ALLOWED_JOB_KINDS)


def test_config_defaults_do_not_point_to_pipeline_profiles() -> None:
    """Ensure config defaults stay on registered model runtimes, not opt-in pipeline profiles."""
    workspace_root = _workspace_root()
    config_path = workspace_root / "src" / "config.py"
    assert config_path.exists(), "config.py does not exist"

    tree = _parse_python_file(config_path)
    registered_models = {
        value
        for spec in list_all_models()
        for value in (spec.alias, spec.model_id)
    }
    pipeline_aliases = {profile.alias for profile in list_all_profiles()}

    for variable_name in ("FUNASR_MODEL_ID", "MLX_MODEL_ID"):
        default_model = _getenv_default_string(tree, variable_name)
        assert default_model not in pipeline_aliases, (
            f"{variable_name} defaults to pipeline profile '{default_model}'. "
            "Default config must stay on a production model runtime, not an opt-in pipeline."
        )
        assert default_model in registered_models, (
            f"{variable_name} default '{default_model}' is not a registered model alias/model_id. "
            "Add it to model_registry.py before making it a config default."
        )


def test_tach_architecture_boundaries() -> None:
    """Layer 1 (Break Build): Module dependency directions must satisfy tach.toml.

    Runs `tach check` as part of the standard pytest suite so boundary violations
    are caught alongside unit and integration tests.
    """
    result = subprocess.run(
        [sys.executable, "-m", "tach", "check"],
        capture_output=True,
        text=True,
        cwd=str(_workspace_root()),
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Architecture boundary violation detected by tach:\n{result.stdout}{result.stderr}\n"
        "Update tach.toml only when the dependency is intentional and design-reviewed."
    )


def test_src_functions_max_nesting_depth() -> None:
    """Layer 2 (Hard Gate): No function in src/ may exceed _MAX_NESTING_DEPTH block nesting levels.

    elif/else chains do not count as deeper nesting — only body blocks do.
    """
    workspace_root = _workspace_root()
    src_dir = workspace_root / "src"

    violations: list[str] = []
    for py_file in sorted(src_dir.rglob("*.py")):
        rel_path = str(py_file.relative_to(workspace_root))
        allowed = _NESTING_DEPTH_ALLOWLIST.get(rel_path, _MAX_NESTING_DEPTH)
        tree = _parse_python_file(py_file)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            depth = _max_block_depth(node.body)
            if depth > allowed:
                violations.append(
                    f"  {rel_path}:{node.lineno} {node.name}() — depth {depth} (max {allowed})"
                )

    assert not violations, (
        f"Block nesting depth exceeded in {len(violations)} function(s):\n"
        + "\n".join(violations)
        + "\nRefactor to reduce nesting or adjust _MAX_NESTING_DEPTH with a design rationale."
    )


def test_ruff_complexity_suppression_allowlist() -> None:
    """Governance: The set of files suppressing Ruff complexity rules must not grow silently.

    Any new per-file-ignores entry covering C901/PLR0911/PLR0912/PLR0915 must be
    added to _APPROVED_COMPLEXITY_SUPPRESSED_FILES in this file with a rationale.
    """
    workspace_root = _workspace_root()
    pyproject_path = workspace_root / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml not found"

    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    per_file_ignores: dict[str, list[str]] = (
        pyproject.get("tool", {}).get("ruff", {}).get("lint", {}).get("per-file-ignores", {})
    )

    files_with_complexity_suppression = {
        pattern
        for pattern, rules in per_file_ignores.items()
        if _COMPLEXITY_RULES & set(rules)
    }

    new_suppressions = files_with_complexity_suppression - _APPROVED_COMPLEXITY_SUPPRESSED_FILES
    assert not new_suppressions, (
        f"Unapproved Ruff complexity suppressions added for: {new_suppressions}.\n"
        "Add the pattern to _APPROVED_COMPLEXITY_SUPPRESSED_FILES in this test with a rationale, "
        "or refactor the code to remove the suppression need."
    )


def test_max_block_depth_elif_vs_else_if() -> None:
    """Regression: elif chains must stay flat; else: if must count as a deeper block.

    Both produce orelse=[ast.If(...)] in the AST.  The col_offset heuristic is the
    only source-position signal that separates them without a CST parser.
    """
    # 1. elif chain: all branches are logically at depth 1
    elif_source = textwrap.dedent("""\
        def f():
            if a:
                pass
            elif b:
                pass
            elif c:
                pass
    """)
    tree = ast.parse(elif_source)
    func_body = tree.body[0].body  # type: ignore[union-attr]
    assert _max_block_depth(func_body) == 1, (
        "elif chain must not inflate depth beyond 1; got deeper"
    )

    # 2. else: if — the nested if is *inside* the else block, so depth must be 2
    else_if_source = textwrap.dedent("""\
        def f():
            if a:
                pass
            else:
                if b:
                    pass
    """)
    tree = ast.parse(else_if_source)
    func_body = tree.body[0].body  # type: ignore[union-attr]
    assert _max_block_depth(func_body) == 2, (
        "else: if must be counted as depth 2 (else body + inner if); got shallower — "
        "check the col_offset heuristic in _max_block_depth"
    )
