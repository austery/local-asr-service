"""Exact interval diarization scoring with one recording-wide speaker mapping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Turn:
    speaker: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if not self.speaker.strip() or not all(map(math.isfinite, (self.start, self.end))):
            raise ValueError("Speaker and timestamps must be valid")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Turn must have nonnegative start and positive duration")


def read_rttm(path: Path, recording: str, duration: float, offset: float = 0) -> list[Turn]:
    turns = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8 or fields[0] != "SPEAKER":
            raise ValueError("Expected RTTM SPEAKER rows")
        if fields[1] != recording:
            continue
        source = Turn(fields[7], float(fields[3]), float(fields[3]) + float(fields[4]))
        start, end = max(0.0, source.start - offset), min(duration, source.end - offset)
        if end > start:
            turns.append(Turn(source.speaker, start, end))
    if not turns:
        raise ValueError("No reference speech overlaps this recording; check ID/offset")
    return turns


def read_uem(
    path: Path, recording: str, duration: float, offset: float = 0
) -> list[tuple[float, float]]:
    intervals = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 4:
            raise ValueError("Expected four UEM fields")
        if fields[0] == recording:
            start, end = (
                max(0.0, float(fields[2]) - offset),
                min(duration, float(fields[3]) - offset),
            )
            if end > start:
                intervals.append((start, end))
    if not intervals:
        raise ValueError("No UEM interval overlaps this recording")
    return intervals


@dataclass(frozen=True)
class Score:
    collar_s: float
    scored_wall_s: float
    reference_speaker_s: float
    missed_speaker_s: float
    false_alarm_speaker_s: float
    confusion_speaker_s: float
    der: float | None
    reference_speakers: int
    hypothesis_speakers: int
    mapping: dict[str, str]


def score_turns(
    reference: list[Turn],
    hypothesis: list[Turn],
    duration: float,
    collar: float = 0.25,
    regions: list[tuple[float, float]] | None = None,
) -> Score:
    """Include overlap; remove centered collars before recording-wide mapping.

    Collar is the total centered width, matching pyannote.metrics; 0.25 s
    excludes 0.125 s on either side of each reference boundary.

    DER is undefined without scored reference speaker-time; retain false alarms
    instead of reporting a perfect score for an empty reference.
    """
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(collar) or collar < 0:
        raise ValueError("Duration must be positive and collar nonnegative")
    regions = [(0.0, duration)] if regions is None else regions
    if not regions or any(not (0 <= a < b <= duration) for a, b in regions):
        raise ValueError("Evaluation regions must be nonempty and within the recording")
    if any(t.end > duration + 1e-6 for t in reference + hypothesis):
        raise ValueError("Turns cannot exceed the recording")
    excluded = (
        [
            (max(0.0, b - collar / 2), min(duration, b + collar / 2))
            for t in reference
            for b in (t.start, t.end)
        ]
        if collar
        else []
    )
    events: dict[float, list[tuple[int, str, int]]] = {0.0: [], duration: []}
    for side, turns in enumerate((reference, hypothesis)):
        for t in turns:
            events.setdefault(t.start, []).append((side, t.speaker, 1))
            events.setdefault(t.end, []).append((side, t.speaker, -1))
    for a, b in regions + excluded:
        events.setdefault(a, [])
        events.setdefault(b, [])
    bounds = sorted(events)
    active: list[dict[str, int]] = [{}, {}]
    cells: list[tuple[float, set[str], set[str]]] = []
    for index, start in enumerate(bounds[:-1]):
        for side, speaker, delta in events[start]:
            active[side][speaker] = active[side].get(speaker, 0) + delta
        end = bounds[index + 1]
        mid = (start + end) / 2
        if not any(a <= mid < b for a, b in regions) or any(a <= mid < b for a, b in excluded):
            continue
        cells.append(
            (
                end - start,
                {k for k, v in active[0].items() if v > 0},
                {k for k, v in active[1].items() if v > 0},
            )
        )
    rlabels = sorted({r for _, rs, _ in cells for r in rs})
    hlabels = sorted({h for _, _, hs in cells for h in hs})
    ri, hi = {v: i for i, v in enumerate(rlabels)}, {v: i for i, v in enumerate(hlabels)}
    agreement = np.zeros((len(rlabels), len(hlabels)))
    for seconds, rs, hs in cells:
        for r in rs:
            for h in hs:
                agreement[ri[r], hi[h]] += seconds
    mapping: dict[str, str] = {}
    if rlabels and hlabels:
        rindices, hindices = linear_sum_assignment(-agreement)
        mapping = {
            hlabels[int(h)]: rlabels[int(r)] for r, h in zip(rindices, hindices, strict=True)
        }
    denominator = miss = false_alarm = confusion = 0.0
    for seconds, rs, hs in cells:
        correct = sum(mapping.get(h) in rs for h in hs)
        denominator += seconds * len(rs)
        miss += seconds * max(0, len(rs) - len(hs))
        false_alarm += seconds * max(0, len(hs) - len(rs))
        confusion += seconds * (min(len(rs), len(hs)) - correct)
    return Score(
        collar,
        sum(x[0] for x in cells),
        denominator,
        miss,
        false_alarm,
        confusion,
        (miss + false_alarm + confusion) / denominator if denominator else None,
        len(rlabels),
        len(hlabels),
        mapping,
    )
