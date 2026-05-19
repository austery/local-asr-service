import ast
import re
from pathlib import Path

from src.core.model_registry import list_all as list_all_models
from src.core.pipeline_registry import list_all_profiles

ALLOWED_JOB_KINDS = ("transcribe", "align", "diarize")


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
