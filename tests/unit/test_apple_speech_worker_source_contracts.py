from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def test_live_runtime_cancels_results_task_on_early_exit() -> None:
    source = (
        WORKSPACE_ROOT
        / "apple-speech-worker"
        / "Sources"
        / "AppleSpeechWorkerCore"
        / "LiveAppleSpeechRuntime.swift"
    ).read_text()

    task_index = source.index("let resultsTask = Task")
    cancel_index = source.index("defer {\n            resultsTask.cancel()\n        }")
    analyze_index = source.index("analyzer.analyzeSequence")

    assert task_index < cancel_index < analyze_index
