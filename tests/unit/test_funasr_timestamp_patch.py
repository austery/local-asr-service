"""Regression tests for the FunASR single-frame CIF timestamp patch.

Trace captured from a real SEACO Paraformer VAD segment (2026-11-21, PureSubs
chunk ``xgAJkGC89rM.normalized.chunk_1.mp3``): a ~120 ms segment holding only
the token "嗯". Both CIF fires land in the last upsampled frame, so FunASR's
``ts_prediction_lfr6_standard`` returns no timestamp, and FunASR 1.4.19's
``sentence_postprocess`` raises instead of returning the transcription.
"""

import funasr.models.paraformer.model as paraformer_model
import funasr.models.seaco_paraformer.model as seaco_model
import pytest
import torch
from funasr.utils import postprocess_utils
from funasr.utils.timestamp_tools import ts_prediction_lfr6_standard as upstream_ts

import src.core.funasr_engine  # noqa: F401  # applies the module-level patch

SINGLE_FRAME_ALPHAS = [0.0335148, 0.0, 0.0, 0.0, 0.2435439, 1.7229414]
SINGLE_FRAME_PEAKS = [0.0335148, 0.0335148, 0.0335148, 0.0335148, 0.2770587, 2.0]

CALL_SITES = [
    pytest.param(seaco_model, id="seaco_paraformer"),
    pytest.param(paraformer_model, id="paraformer"),
]


@pytest.mark.parametrize("module", CALL_SITES)
def test_single_frame_double_fire_keeps_token_with_segment_timestamp(module: object) -> None:
    alphas = torch.tensor(SINGLE_FRAME_ALPHAS)
    peaks = torch.tensor(SINGLE_FRAME_PEAKS)

    _, timestamps = module.ts_prediction_lfr6_standard(  # type: ignore[attr-defined]
        alphas, peaks, ["嗯"], vad_offset=0
    )
    text, sentence_ts, _ = postprocess_utils.sentence_postprocess(["嗯"], timestamps)

    assert text == "嗯"
    # 6 upsampled frames x 20 ms = the 120 ms segment.
    assert sentence_ts == [[0, 120]]


@pytest.mark.parametrize("module", CALL_SITES)
def test_filled_timestamps_include_vad_offset(module: object) -> None:
    _, timestamps = module.ts_prediction_lfr6_standard(  # type: ignore[attr-defined]
        torch.tensor(SINGLE_FRAME_ALPHAS),
        torch.tensor(SINGLE_FRAME_PEAKS),
        ["嗯"],
        vad_offset=5000,
    )

    assert timestamps == [[5000, 5120]]


def test_missing_tokens_share_remaining_segment_after_known_timestamps() -> None:
    # Two tokens, one fire short: the second token falls in the final frame.
    alphas = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    peaks = torch.zeros_like(alphas)

    _, upstream = upstream_ts(alphas.clone(), peaks.clone(), ["你", "好"])
    _, timestamps = seaco_model.ts_prediction_lfr6_standard(alphas, peaks, ["你", "好"])

    assert len(upstream) < 2, "fixture must reproduce the upstream shortfall"
    assert len(timestamps) == 2
    assert timestamps[: len(upstream)] == upstream
    starts_ends = [value for pair in timestamps for value in pair]
    assert starts_ends == sorted(starts_ends)
    assert timestamps[-1][1] == 9 * 20


def test_complete_timestamps_are_returned_unchanged() -> None:
    alphas = torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0])
    peaks = torch.zeros_like(alphas)

    expected = upstream_ts(alphas.clone(), peaks.clone(), ["你", "好"])
    actual = seaco_model.ts_prediction_lfr6_standard(alphas, peaks, ["你", "好"])

    assert len(expected[1]) == 2
    assert actual == expected
