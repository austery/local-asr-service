"""Regression tests for the FunASR single-frame CIF timestamp patch.

Trace captured from a real SEACO Paraformer VAD segment (2026-11-21, PureSubs
chunk ``xgAJkGC89rM.normalized.chunk_1.mp3``): a ~120 ms segment holding only
the token "嗯". Both CIF fires land in the last upsampled frame, so FunASR's
``ts_prediction_lfr6_standard`` returns no timestamp, and FunASR 1.4.19's
``sentence_postprocess`` raises instead of returning the transcription.
"""

import importlib
import inspect

import funasr.models.seaco_paraformer.model as seaco_model
import pytest
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
    # The lost fire was in the last of 6 upsampled frames (20 ms each).
    assert sentence_ts == [[100, 120]]


def test_filled_timestamps_include_vad_offset() -> None:
    _, timestamps = seaco_model.ts_prediction_lfr6_standard(
        torch.tensor(SINGLE_FRAME_ALPHAS),
        torch.tensor(SINGLE_FRAME_PEAKS),
        ["嗯"],
        vad_offset=5000,
    )

    assert timestamps == [[5100, 5120]]


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
    # rest as <sil>; the missing token takes the last frame, not the silence.
    alphas = torch.zeros(16)
    alphas[2], alphas[15] = 1.0, 2.0
    peaks = torch.zeros_like(alphas)

    _, upstream = upstream_ts(alphas.clone(), peaks.clone(), ["你", "好"])
    _, timestamps = seaco_model.ts_prediction_lfr6_standard(alphas, peaks, ["你", "好"])

    assert upstream == [[10, 250]], "fixture must reproduce the upstream shortfall"
    assert timestamps == [[10, 250], [300, 320]]


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

    assert timestamps == [[5100, 5120]]


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


def test_merged_share_borrows_only_needed_leading_silence() -> None:
    # 7 frames, 4 tokens, fires bunched at the end: each token gets one frame.
    alphas = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0])
    peaks = torch.zeros_like(alphas)

    _, upstream = upstream_ts(alphas.clone(), peaks.clone(), list("abcd"))
    _, timestamps = seaco_model.ts_prediction_lfr6_standard(alphas, peaks, list("abcd"))

    assert upstream == [[70, 140]], "fixture must reproduce the upstream shortfall"
    assert timestamps == [[60, 80], [80, 100], [100, 120], [120, 140]]


@pytest.mark.parametrize(
    ("char_list", "upsample_rate", "expected"),
    [
        pytest.param(["嗯", "</s>"], 3, [[100, 120]], id="trailing-eos"),
        pytest.param(["嗯"], 6, [[50, 60]], id="upsample-rate-6"),
        pytest.param([], 3, [], id="empty-token-list"),
    ],
)
def test_fill_handles_upstream_token_and_rate_conventions(
    char_list: list[str], upsample_rate: int, expected: list[list[int]]
) -> None:
    _, timestamps = seaco_model.ts_prediction_lfr6_standard(
        torch.tensor(SINGLE_FRAME_ALPHAS),
        torch.tensor(SINGLE_FRAME_PEAKS),
        list(char_list),
        upsample_rate=upsample_rate,
    )

    assert timestamps == expected


def test_complete_timestamps_accept_any_vad_offset_upstream_accepts() -> None:
    alphas = torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0])
    peaks = torch.zeros_like(alphas)

    expected = upstream_ts(alphas.clone(), peaks.clone(), ["你", "好"], vad_offset=None)
    actual = seaco_model.ts_prediction_lfr6_standard(alphas, peaks, ["你", "好"], vad_offset=None)

    assert actual == expected


def test_patch_is_skipped_when_upstream_drops_a_required_parameter(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from funasr.utils import timestamp_tools

    from src.core import funasr_engine

    def renamed_upstream(  # simulates a FunASR release without upsample_rate
        us_alphas: torch.Tensor, us_peaks: torch.Tensor, char_list: list[str], vad_offset: float = 0.0
    ) -> tuple[str, list[list[int]]]:
        return "", []

    monkeypatch.setattr(timestamp_tools, "ts_prediction_lfr6_standard", renamed_upstream)
    monkeypatch.setattr(seaco_model, "ts_prediction_lfr6_standard", renamed_upstream)
    try:
        with caplog.at_level("WARNING"):
            importlib.reload(funasr_engine)
        assert seaco_model.ts_prediction_lfr6_standard is renamed_upstream
        assert "single-frame CIF timestamp patch was NOT applied" in caplog.text
    finally:
        monkeypatch.undo()
        importlib.reload(funasr_engine)
    assert seaco_model.ts_prediction_lfr6_standard is funasr_engine._patched_ts_prediction_lfr6_standard


def test_trailing_untimed_token_does_not_trigger_fill() -> None:
    # sentence_postprocess drops a trailing <unk>, so upstream output is enough.
    alphas = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    peaks = torch.zeros_like(alphas)

    expected = upstream_ts(alphas.clone(), peaks.clone(), ["你", "<unk>"])
    actual = seaco_model.ts_prediction_lfr6_standard(alphas, peaks, ["你", "<unk>"])

    assert actual == expected
    assert postprocess_utils.sentence_postprocess(["你", "<unk>"], actual[1])[:2] == (
        "你",
        [[10, 180]],
    )


def test_long_segment_without_timestamps_fills_at_segment_end() -> None:
    # 10 s segment; all weight unfired in the last frame.
    alphas = torch.zeros(500)
    alphas[499] = 4.0
    peaks = torch.zeros_like(alphas)

    _, upstream = upstream_ts(alphas.clone(), peaks.clone(), list("abc"))
    _, timestamps = seaco_model.ts_prediction_lfr6_standard(alphas, peaks, list("abc"))

    assert upstream == [], "fixture must reproduce the upstream shortfall"
    assert timestamps == [[9940, 9960], [9960, 9980], [9980, 10000]]


def test_long_trailing_silence_is_not_given_to_filled_token() -> None:
    alphas = torch.zeros(500)
    alphas[2], alphas[499] = 1.0, 2.0
    peaks = torch.zeros_like(alphas)

    _, upstream = upstream_ts(alphas.clone(), peaks.clone(), ["你", "好"])
    _, timestamps = seaco_model.ts_prediction_lfr6_standard(alphas, peaks, ["你", "好"])

    assert upstream == [[10, 250]], "fixture must reproduce the upstream shortfall"
    assert timestamps == [[10, 250], [9980, 10000]]


def _reload_engine_with_upstream(
    monkeypatch: pytest.MonkeyPatch, fake: object, seaco_binding: object
) -> object:
    from funasr.utils import timestamp_tools

    from src.core import funasr_engine

    monkeypatch.setattr(timestamp_tools, "ts_prediction_lfr6_standard", fake)
    monkeypatch.setattr(seaco_model, "ts_prediction_lfr6_standard", seaco_binding)
    importlib.reload(funasr_engine)
    return funasr_engine


def _restore_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core import funasr_engine

    monkeypatch.undo()
    importlib.reload(funasr_engine)
    assert (
        seaco_model.ts_prediction_lfr6_standard
        is funasr_engine._patched_ts_prediction_lfr6_standard
    )


def test_patch_survives_upstream_adding_a_positional_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def extended_upstream(  # simulates a FunASR release with a new parameter
        us_alphas: torch.Tensor,
        us_peaks: torch.Tensor,
        frame_hint: int,
        char_list: list[str],
        vad_offset: float = 0.0,
        upsample_rate: int = 3,
    ) -> tuple[str, list[list[int]]]:
        return upstream_ts(us_alphas, us_peaks, char_list, vad_offset, upsample_rate=upsample_rate)

    try:
        _reload_engine_with_upstream(monkeypatch, extended_upstream, extended_upstream)
        _, timestamps = seaco_model.ts_prediction_lfr6_standard(
            torch.tensor(SINGLE_FRAME_ALPHAS), torch.tensor(SINGLE_FRAME_PEAKS), 6, ["嗯"]
        )
        assert timestamps == [[100, 120]]
    finally:
        _restore_engine(monkeypatch)


def test_unreadable_arguments_return_upstream_result(monkeypatch: pytest.MonkeyPatch) -> None:
    # A decorated upstream whose reported signature no longer matches its calls.
    def decorated_upstream(*args: object, **kwargs: object) -> tuple[str, list[list[int]]]:
        kwargs.pop("calibration", None)
        return upstream_ts(*args, **kwargs)  # type: ignore[arg-type]

    decorated_upstream.__signature__ = inspect.signature(upstream_ts)  # type: ignore[attr-defined]
    alphas = torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0])
    try:
        _reload_engine_with_upstream(monkeypatch, decorated_upstream, decorated_upstream)
        actual = seaco_model.ts_prediction_lfr6_standard(
            alphas, torch.zeros_like(alphas), ["你", "好"], calibration=1
        )
        assert actual == upstream_ts(alphas.clone(), torch.zeros_like(alphas), ["你", "好"])
    finally:
        _restore_engine(monkeypatch)


def test_patch_is_skipped_when_seaco_no_longer_imports_the_function(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from src.core import funasr_engine

    monkeypatch.delattr(seaco_model, "ts_prediction_lfr6_standard")
    try:
        with caplog.at_level("WARNING"):
            importlib.reload(funasr_engine)
        assert not hasattr(seaco_model, "ts_prediction_lfr6_standard")
        assert "single-frame CIF timestamp patch was NOT applied" in caplog.text
    finally:
        _restore_engine(monkeypatch)
