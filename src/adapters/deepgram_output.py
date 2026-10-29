"""Fail-closed normalization of Deepgram's batch diarized response."""

from itertools import groupby

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.core.remote_port import (
    RemoteSegment,
    RemoteTranscript,
    RemoteTranscriptionError,
    RemoteWord,
)


class _Timing(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    speaker: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def positive_interval(self) -> "_Timing":
        if self.end <= self.start:
            raise ValueError("Invalid interval")
        return self


class _Word(_Timing):
    word: str = Field(min_length=1)
    punctuated_word: str | None = None

    def normalized(self) -> RemoteWord:
        if not self.word.strip():
            raise ValueError("Empty word")
        return RemoteWord(f"Speaker {self.speaker}", self.start, self.end, self.punctuated_word or self.word)


class _Utterance(_Timing):
    transcript: str = Field(min_length=1)
    words: list[_Word] = Field(min_length=1)

    def normalized(self, duration: float) -> tuple[RemoteSegment, ...]:
        if not self.transcript.strip() or self.end > duration:
            raise ValueError("Invalid utterance")
        for word in self.words:
            if word.start < self.start or word.end > self.end:
                raise ValueError("Word outside utterance")
        runs = tuple(
            tuple(run)
            for _, run in groupby((word.normalized() for word in self.words), key=lambda w: w.speaker)
        )
        if len(runs) == 1:
            # Word identity wins even when the utterance header disagrees.
            return (RemoteSegment(runs[0][0].speaker, self.start, self.end, self.transcript, runs[0]),)
        # Preserve each token verbatim (punctuated_word or word), joined by one
        # space. Do not infer punctuation from the utterance transcript.
        return tuple(
            RemoteSegment(
                run[0].speaker, min(w.start for w in run), max(w.end for w in run),
                " ".join(w.text for w in run), run,
            )
            for run in runs
        )


class _Alternative(BaseModel):
    model_config = ConfigDict(strict=True)
    transcript: str
    words: list[_Word]


class _Channel(BaseModel):
    alternatives: list[_Alternative] = Field(min_length=1, max_length=1)


class _Results(BaseModel):
    channels: list[_Channel] = Field(min_length=1, max_length=1)
    utterances: list[_Utterance] = Field(default_factory=list)


class _Metadata(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    duration: float = Field(ge=0)


class _Response(BaseModel):
    metadata: _Metadata
    results: _Results


def normalize_deepgram(data: object, language: str) -> RemoteTranscript:
    try:
        response = _Response.model_validate(data)
        alternative = response.results.channels[0].alternatives[0]
        segments = tuple(
            segment
            for utterance in response.results.utterances
            for segment in utterance.normalized(response.metadata.duration)
        )
        words = tuple(w.normalized() for w in alternative.words)
        if any(w.end > response.metadata.duration for w in words):
            raise ValueError("Word exceeds duration")
        speech = bool(alternative.transcript.strip())
        if speech != bool(segments) or speech != bool(words):
            raise ValueError("Speech requires labeled words and utterances")
        return RemoteTranscript(alternative.transcript, response.metadata.duration, language, segments)
    except (ValidationError, ValueError):
        raise RemoteTranscriptionError(502, "Deepgram returned invalid diarized output; do not automatically resend.") from None
