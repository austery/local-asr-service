"""Reference-label scoring behavior; no real model imports or inference."""

from pathlib import Path

import pytest

from benchmarks.diarization_metrics import Turn, read_rttm, read_uem, score_turns


def test_global_permutation_and_overlap_are_scored_correctly() -> None:
    reference = [Turn("A", 0, 2), Turn("B", 1, 3)]
    hypothesis = [Turn("7", 0, 2), Turn("3", 1, 3)]
    result = score_turns(reference, hypothesis, 3, collar=0)
    assert result.der == 0
    assert result.reference_speaker_s == 4
    assert result.mapping == {"7": "A", "3": "B"}


def test_missing_overlap_counts_as_missed_speaker_time() -> None:
    result = score_turns([Turn("A", 0, 2), Turn("B", 1, 3)], [Turn("x", 0, 3)], 3, collar=0)
    assert result.missed_speaker_s == 1
    assert result.confusion_speaker_s == 1
    assert result.der == 0.5


def test_unmatched_extra_speaker_is_false_alarm() -> None:
    result = score_turns([Turn("A", 0, 2)], [Turn("x", 0, 2), Turn("y", 0, 2)], 2, collar=0)
    assert result.false_alarm_speaker_s == 2
    assert result.der == 1


def test_silence_does_not_report_a_perfect_der() -> None:
    result = score_turns([], [Turn("x", 1, 2)], 3, collar=0)
    assert result.der is None
    assert result.false_alarm_speaker_s == 1


def test_label_switches_cannot_be_hidden_by_per_chunk_remapping() -> None:
    reference = [Turn("A", 0, 2), Turn("B", 2, 4)]
    hypothesis = [Turn("x", 0, 1), Turn("y", 1, 2), Turn("x", 2, 3), Turn("y", 3, 4)]
    assert score_turns(reference, hypothesis, 4, collar=0).der == 0.5


def test_reference_collars_remove_boundary_uncertainty_only() -> None:
    reference = [Turn("A", 0, 1), Turn("B", 1, 2)]
    hypothesis = [Turn("x", 0, 1.1), Turn("y", 1.1, 2)]
    assert score_turns(reference, hypothesis, 2, collar=0).der == pytest.approx(0.05)
    result = score_turns(reference, hypothesis, 2, collar=0.25)
    assert result.der == 0
    assert result.scored_wall_s == 1.5


def test_uem_regions_exclude_unannotated_audio() -> None:
    result = score_turns([Turn("A", 0, 2)], [Turn("x", 0, 1)], 2, collar=0, regions=[(0, 1)])
    assert result.der == 0
    assert result.reference_speaker_s == 1


def test_duplicate_same_speaker_intervals_are_not_double_counted() -> None:
    result = score_turns([Turn("A", 0, 2), Turn("A", 1, 3)], [Turn("x", 0, 3)], 3, collar=0)
    assert result.reference_speaker_s == 3
    assert result.der == 0


@pytest.mark.parametrize("start,end", [(1, 1), (-1, 2), (0, float("nan")), (0, float("inf"))])
def test_invalid_turns_fail_loudly(start: float, end: float) -> None:
    with pytest.raises(ValueError):
        Turn("A", start, end)


def test_empty_scoring_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="regions"):
        score_turns([], [], 2, regions=[])


def test_rttm_and_uem_are_clipped_using_the_same_offset(tmp_path: Path) -> None:
    ref = tmp_path / "ref.rttm"
    ref.write_text("SPEAKER meeting 1 8 5 <NA> <NA> A <NA> <NA>\n")
    uem = tmp_path / "ref.uem"
    uem.write_text("meeting 1 9 15\n")
    assert read_rttm(ref, "meeting", 4, offset=10) == [Turn("A", 0, 3)]
    assert read_uem(uem, "meeting", 4, offset=10) == [(0, 4)]
    with pytest.raises(ValueError, match="No reference"):
        read_rttm(ref, "wrong", 4, offset=10)


@pytest.mark.parametrize(
    "duration",
    [0, -1, float("nan"), float("inf"), float("-inf")],
    ids=["zero", "negative", "nan", "positive-infinity", "negative-infinity"],
)
def test_uem_rejects_invalid_duration_before_clipping(tmp_path: Path, duration: float) -> None:
    uem = tmp_path / "ref.uem"
    uem.write_text("meeting 1 9 15\n")
    with pytest.raises(ValueError, match="UEM duration must be positive"):
        read_uem(uem, "meeting", duration=duration, offset=10)


@pytest.mark.parametrize(
    "offset",
    [-1, float("nan"), float("inf"), float("-inf")],
    ids=["negative", "nan", "positive-infinity", "negative-infinity"],
)
def test_uem_rejects_invalid_offset_before_clipping(tmp_path: Path, offset: float) -> None:
    uem = tmp_path / "ref.uem"
    uem.write_text("meeting 1 9 15\n")
    with pytest.raises(ValueError, match="offset nonnegative and finite"):
        read_uem(uem, "meeting", duration=4, offset=offset)


@pytest.mark.parametrize(
    "start,end",
    [
        ("nan", "2"),
        ("0", "nan"),
        ("-inf", "2"),
        ("0", "inf"),
        ("inf", "2"),
        ("0", "-inf"),
        ("-1", "2"),
        ("2", "1"),
        ("1", "1"),
    ],
)
def test_uem_rejects_invalid_source_coordinates_before_clipping(
    tmp_path: Path, start: str, end: str
) -> None:
    uem = tmp_path / "bad.uem"
    uem.write_text(f"clip 1 {start} {end}\n")
    with pytest.raises(ValueError, match="UEM coordinates"):
        read_uem(uem, "clip", duration=4)
