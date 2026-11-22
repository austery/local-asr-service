"""Regression tests for the FunASR single-frame CIF timestamp patch.

Trace captured from a real SEACO Paraformer VAD segment (2026-11-21, PureSubs
chunk ``xgAJkGC89rM.normalized.chunk_1.mp3``): a ~120 ms segment holding only
the token "嗯". Both CIF fires land in the last upsampled frame, so FunASR's
``ts_prediction_lfr6_standard`` returns no timestamp, and FunASR 1.4.19's
``sentence_postprocess`` raises instead of returning the transcription.
"""

import funasr.models.seaco_paraformer.model as seaco_model
import torch
from funasr.utils import postprocess_utils
from funasr.utils.timestamp_tools import ts_prediction_lfr6_standard as upstream_ts

import src.core.funasr_engine  # noqa: F401  # applies the module-level patch

SINGLE_FRAME_ALPHAS = [0.0335148, 0.0, 0.0, 0.0, 0.2435439, 1.7229414]
SINGLE_FRAME_PEAKS = [0.0335148, 0.0335148, 0.0335148, 0.0335148, 0.2770587, 2.0]


def test_single_frame_double_fire_keeps_token_with_segment_timestamp() -> None:
    alphas = torch.tensor(SINGLE_FRAME_ALPHAS)
    peaks = torch.tensor(SINGLE_FRAME_PEAKS)

    _, timestamps = seaco_model.ts_prediction_lfr6_standard(
        alphas, peaks, ["嗯"], vad_offset=0
    )
    text, sentence_ts, _ = postprocess_utils.sentence_postprocess(["嗯"], timestamps)

    assert text == "嗯"
    # 6 upsampled frames x 20 ms = the 120 ms segment.
    assert sentence_ts == [[0, 120]]


def test_filled_timestamps_include_vad_offset() -> None:
    _, timestamps = seaco_model.ts_prediction_lfr6_standard(
        torch.tensor(SINGLE_FRAME_ALPHAS),
        torch.tensor(SINGLE_FRAME_PEAKS),
        ["嗯"],
        vad_offset=5000,
    )

    assert timestamps == [[5000, 5120]]


def test_missing_token_splits_last_span_when_upstream_reached_segment_end() -> None:
    # Two tokens, one fire short: the second token falls in the final frame, and
    # upstream stretches "你" to the segment end (180 ms), leaving no room.
    alphas = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    peaks = torch.zeros_like(alphas)

    _, upstream = upstream_ts(alphas.clone(), peaks.clone(), ["你", "好"])
    _, timestamps = seaco_model.ts_prediction_lfr6_standard(alphas, peaks, ["你", "好"])

    assert upstream == [[10, 180]], "fixture must reproduce the upstream shortfall"
    assert timestamps == [[10, 95], [95, 180]]
    assert all(end > start for start, end in timestamps)


def test_missing_token_uses_remaining_segment_after_trailing_silence() -> None:
    # 16 frames: upstream caps "你" at MAX_TOKEN_DURATION (250 ms) and marks the
    # rest as <sil>, so the missing token fits in the 250-320 ms tail.
    alphas = torch.zeros(16)
    alphas[2], alphas[15] = 1.0, 2.0
    peaks = torch.zeros_like(alphas)

    _, upstream = upstream_ts(alphas.clone(), peaks.clone(), ["你", "好"])
    _, timestamps = seaco_model.ts_prediction_lfr6_standard(alphas, peaks, ["你", "好"])

    assert upstream == [[10, 250]], "fixture must reproduce the upstream shortfall"
    assert timestamps == [[10, 250], [250, 320]]


def test_complete_timestamps_are_returned_unchanged() -> None:
    alphas = torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0])
    peaks = torch.zeros_like(alphas)

    expected = upstream_ts(alphas.clone(), peaks.clone(), ["你", "好"])
    actual = seaco_model.ts_prediction_lfr6_standard(alphas, peaks, ["你", "好"])

    assert len(expected[1]) == 2
    assert actual == expected


def test_positional_arguments_are_forwarded_like_upstream() -> None:
    args = (torch.tensor(SINGLE_FRAME_ALPHAS), torch.tensor(SINGLE_FRAME_PEAKS), ["嗯"], 5000)

    _, timestamps = seaco_model.ts_prediction_lfr6_standard(*args)

    assert timestamps == [[5000, 5120]]


def test_rounding_gap_with_vad_offset_still_splits_last_span() -> None:
    # Upstream truncates its end to 199 ms while the segment ends at 200 ms.
    alphas = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    peaks = torch.zeros_like(alphas)

    _, upstream = upstream_ts(alphas.clone(), peaks.clone(), ["你", "好"], vad_offset=20)
    _, timestamps = seaco_model.ts_prediction_lfr6_standard(
        alphas, peaks, ["你", "好"], vad_offset=20
    )

    assert upstream == [[30, 199]], "fixture must reproduce the upstream 1 ms gap"
    assert timestamps == [[30, 115], [115, 200]]


def test_long_segment_rounding_gap_with_two_missing_tokens() -> None:
    # Production path (vad_offset=0): ~16 s segment, final-frame shortfall of two.
    alphas = torch.zeros(803)
    alphas[799], alphas[802] = 1.0, 3.0
    peaks = torch.zeros_like(alphas)
    tokens = ["你", "好", "吗"]

    _, upstream = upstream_ts(alphas.clone(), peaks.clone(), list(tokens))
    _, timestamps = seaco_model.ts_prediction_lfr6_standard(alphas, peaks, list(tokens))

    assert upstream == [[15950, 16059]], "fixture must reproduce the upstream gap"
    assert len(timestamps) == 3
    assert timestamps[0][0] == 15950
    assert all(end - start >= 20 for start, end in timestamps)
    assert timestamps[-1][1] == 16060


def test_split_never_starts_before_segment_start() -> None:
    # Upstream's force_time_shift puts the first token at -10 ms.
    alphas = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 2.0])
    peaks = torch.zeros_like(alphas)

    _, upstream = upstream_ts(alphas.clone(), peaks.clone(), ["你", "好"])
    _, timestamps = seaco_model.ts_prediction_lfr6_standard(alphas, peaks, ["你", "好"])

    assert upstream == [[-10, 120]], "fixture must reproduce the negative upstream start"
    assert timestamps == [[0, 60], [60, 120]]
