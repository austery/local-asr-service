import ast
import re
from pathlib import Path

from src.core.model_registry import list_all as list_all_models
from src.core.pipeline_registry import list_all_profiles


def test_models_md_documents_all_aliases() -> None:
    """Verify that all model aliases and pipeline profiles in the registry are documented in MODELS.md."""
    workspace_root = Path(__file__).parent.parent.parent
    models_md_path = workspace_root / "MODELS.md"
    assert models_md_path.exists(), "MODELS.md does not exist at project root"

    with open(models_md_path, "r", encoding="utf-8") as f:
        content = f.read()

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
    workspace_root = Path(__file__).parent.parent.parent
    registry_path = workspace_root / "src" / "core" / "pipeline_registry.py"
    assert registry_path.exists(), "pipeline_registry.py does not exist"

    with open(registry_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(registry_path))

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
    workspace_root = Path(__file__).parent.parent.parent

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

    unexpected_adapters = current_adapters_files - allowed_adapters_files
    assert not unexpected_adapters, (
        f"Unexpected files found in src/adapters: {unexpected_adapters}. "
        f"To add a new processing helper, you must register it in this test."
    )


def test_transcription_service_job_domains() -> None:
    """Verify that the job kind domains in TranscriptionService are strictly limited to transcribe/align/diarize."""
    workspace_root = Path(__file__).parent.parent.parent
    service_path = workspace_root / "src" / "services" / "transcription.py"
    assert service_path.exists(), "transcription.py does not exist"

    with open(service_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(service_path))

    # We want to scan the ast to find all methods or calls that parameterize job kind
    # and check they only pass "transcribe", "align", or "diarize".
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check calls to '_submit_worker_job' or '_enqueue_worker_job'
            # and verify keyword/positional arguments for job_kind
            func_name = ""
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id

            if func_name in ("_submit_worker_job", "_enqueue_worker_job"):
                # Find the job_kind argument
                job_kind_val = None
                # First check keywords
                for kw in node.keywords:
                    if kw.arg == "job_kind" and isinstance(kw.value, ast.Constant):
                        job_kind_val = kw.value.value

                if job_kind_val is not None:
                    assert job_kind_val in ("transcribe", "align", "diarize"), (
                        f"Disallowed job_kind '{job_kind_val}' passed to {func_name}. "
                        f"Only 'transcribe', 'align', and 'diarize' are allowed."
                    )
