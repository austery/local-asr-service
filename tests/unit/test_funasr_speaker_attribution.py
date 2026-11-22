"""Speaker attribution uses upstream FunASR distribute_spk.

The engine used to replace distribute_spk with a local copy that filtered None
speaker timestamps and picked the single largest overlap. FunASR 1.4.19 sums
overlap per speaker, and its speaker segments cannot be None, so the engine
now leaves both upstream bindings in place.
"""

from pathlib import Path

import funasr.models.campplus.utils as campplus_utils
from funasr.auto import auto_model

import src.core.funasr_engine  # noqa: F401  (import-time side effects under test)


def test_engine_import_keeps_upstream_distribute_spk_bindings() -> None:
    upstream = campplus_utils.distribute_spk

    assert upstream.__module__ == "funasr.models.campplus.utils"
    # functools.wraps copies __module__; the code object keeps its source file.
    assert Path(upstream.__code__.co_filename).as_posix().endswith(
        "funasr/models/campplus/utils.py"
    )
    assert auto_model.distribute_spk is upstream


def test_sentence_goes_to_speaker_with_largest_total_overlap() -> None:
    # Speaker 0 speaks 0-400 ms and 900-1300 ms (800 ms total); speaker 1
    # speaks 400-900 ms (500 ms, the largest single segment).
    sentences: list[dict[str, object]] = [{"start": 0, "end": 1300}]
    speaker_segments = [[0.0, 0.4, 0], [0.4, 0.9, 1], [0.9, 1.3, 0]]

    auto_model.distribute_spk(sentences, speaker_segments)

    assert sentences[0]["spk"] == 0


def test_sentence_without_overlap_defaults_to_speaker_zero() -> None:
    sentences: list[dict[str, object]] = [{"start": 5000, "end": 6000}]

    auto_model.distribute_spk(sentences, [[0.0, 1.0, 3]])

    assert sentences[0]["spk"] == 0


def test_equal_total_overlap_goes_to_first_overlapping_speaker() -> None:
    # Speaker 0: 0-100 ms and 400-600 ms (300 ms total); speaker 1: 100-400 ms
    # (300 ms). The largest single overlap is speaker 1, but upstream keeps the
    # first overlapping speaker on a tie.
    sentences: list[dict[str, object]] = [{"start": 0, "end": 600}]

    auto_model.distribute_spk(sentences, [[0.0, 0.1, 0], [0.1, 0.4, 1], [0.4, 0.6, 0]])

    assert sentences[0]["spk"] == 0
