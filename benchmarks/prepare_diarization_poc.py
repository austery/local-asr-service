"""Prepare a small public AMI test subset and deterministic control recordings.

AMI audio: CC BY 4.0, University of Edinburgh AMI Corpus Mirror.
RTTM setup: pyannote/AMI-diarization-setup, derived from AMI manual annotations
v1.6.2. This tool fetches individual meetings, never a full dataset archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
import wave
from pathlib import Path

import numpy as np
from scipy.io import wavfile

REVISION = "67c2d539286e89f68952d5dcf83912bd9f01dfae"
MEETINGS = ("ES2004a", "ES2004b", "IS1009a")


def fetch(url: str, destination: Path) -> None:
    if destination.exists():
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"Inspect the previous interrupted download: {partial}")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        total = 0
        while block := response.read(1024 * 1024):
            total += len(block)
            if total > 128 * 1024**2:
                raise ValueError("Individual download exceeded the 128 MiB corpus budget")
            out.write(block)
    partial.rename(destination)


def clip(source: Path, output: Path, offset: float, seconds: float) -> None:
    if output.exists():
        raise FileExistsError(f"Preserve existing derived audio: {output}")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(offset),
            "-i",
            str(source),
            "-t",
            str(seconds),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    media, refs = args.output / "media", args.output / "references"
    media.mkdir(parents=True, exist_ok=True)
    refs.mkdir(parents=True, exist_ok=True)
    if (args.output / "public-corpus.json").exists():
        raise FileExistsError("Choose a new corpus directory to retain the previous manifest")
    for target in (
        media / "ami-a-180.wav",
        media / "ami-b-300.wav",
        media / "silence60.wav",
        media / "empty.wav",
        media / "capacity8-return.wav",
        refs / "capacity8-return.rttm",
    ):
        if target.exists():
            raise FileExistsError(f"Preserve existing derived evidence: {target}")
    files = []
    for meeting in MEETINGS:
        urls = [
            (
                f"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/audio/{meeting}.Mix-Headset.wav",
                media / f"{meeting}.wav",
            ),
            (
                f"https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/{REVISION}/only_words/rttms/test/{meeting}.rttm",
                refs / f"{meeting}.rttm",
            ),
        ]
        if meeting != "IS1009a":
            urls.append(
                (
                    f"https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/{REVISION}/uems/test/{meeting}.uem",
                    refs / f"{meeting}.uem",
                )
            )
        for url, path in urls:
            fetch(url, path)
            files.append(
                {
                    "url": url,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    clip(media / "ES2004a.wav", media / "ami-a-180.wav", 0, 180)
    clip(media / "ES2004b.wav", media / "ami-b-300.wav", 300, 300)
    wavfile.write(media / "silence60.wav", 16000, np.zeros(60 * 16000, dtype=np.int16))
    wavfile.write(media / "empty.wav", 16000, np.zeros(0, dtype=np.int16))
    # A capacity stress test, explicitly constructed rather than a natural meeting.
    samples = []
    annotations = []
    cursor = 0.0
    for meeting, seconds in [("ES2004a", 360), ("IS1009a", 240), ("ES2004a", 90)]:
        with wave.open(str(media / f"{meeting}.wav"), "rb") as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (
                1,
                2,
                16000,
            ):
                raise ValueError("AMI mix format changed; inspect before construction")
            samples.append(np.frombuffer(source.readframes(round(seconds * 16000)), dtype="<i2"))
        for row in (refs / f"{meeting}.rttm").read_text().splitlines():
            fields = row.split()
            start, end = float(fields[3]), float(fields[3]) + float(fields[4])
            end = min(seconds, end)
            if end > start:
                fields[1] = "capacity8-return"
                fields[3], fields[4] = str(start + cursor), str(end - start)
                annotations.append(" ".join(fields))
        cursor += seconds
    if len({row.split()[7] for row in annotations}) != 8:
        raise ValueError("Capacity reference must contain exactly eight distinct speakers")
    wavfile.write(media / "capacity8-return.wav", 16000, np.concatenate(samples))
    (refs / "capacity8-return.rttm").write_text("\n".join(annotations) + "\n")
    (args.output / "public-corpus.json").write_text(
        json.dumps(
            {
                "reference_revision": REVISION,
                "annotation": "only_words derived from AMI manual v1.6.2",
                "audio_license": "CC-BY-4.0",
                "reference_setup_license": "Apache-2.0; retain AMI attribution",
                "split": "AMI test",
                "files": files,
                "capacity": "Synthetic concatenation of two distinct four-speaker groups followed by a returning first group; not natural eight-person acceptance.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
