"""Check public OpenAPI instructions without starting a service or worker."""

from typing import cast

import pytest
from fastapi import FastAPI

from src.api.routes import router


def mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


@pytest.fixture
def schema() -> dict[str, object]:
    app = FastAPI()
    app.include_router(router)
    return mapping(app.openapi())


def test_deepgram_operation_instructions(schema: dict[str, object]) -> None:
    paths = mapping(schema["paths"])
    operation = mapping(mapping(paths["/v1/audio/transcriptions"])["post"])
    description = operation["description"]
    assert isinstance(description, str)
    for instruction in (
        "model=deepgram-nova-3-diarize", "language=en", "language=zh",
        "response_format=verbose_json", "DEEPGRAM_ENABLED=true", "DEEPGRAM_API_KEY",
        "HOST=127.0.0.1", "authenticated proxy", "Never enter the key",
        "merged code and current docs", "server restart", "browser refresh",
        "Keep startup `MODEL_ID` local", "GET /v1/models", "configured",
        "requestable", "unavailable_reason", "503", "without a paid upload", "400",
        "200 MiB", "lower configured", "wav/mp3/m4a/mp4/flac/ogg/webm",
        "no audio-minute cap", "Processing timeouts are not recording-duration limits",
        "single-channel provider response", "MODELS.md", "No automatic retry, fallback",
        "not a resident model switch", "upload audio externally", "spend credits",
    ):
        assert instruction in description
    steps = (
        "4. In Swagger", "**Try it out**", "5. Click **Choose file**",
        "6. Set **model**", "7. Set **language**", "8. Set **response_format**",
        "9. Click **Execute**",
    )
    positions = [description.index(step) for step in steps]
    assert positions == sorted(positions)


def test_form_instructions_and_existing_defaults(schema: dict[str, object]) -> None:
    paths = mapping(schema["paths"])
    operation = mapping(mapping(paths["/v1/audio/transcriptions"])["post"])
    content = mapping(mapping(operation["requestBody"])["content"])
    body_ref = mapping(mapping(content["multipart/form-data"])["schema"])["$ref"]
    assert isinstance(body_ref, str)
    schemas = mapping(mapping(schema["components"])["schemas"])
    body = mapping(schemas[body_ref.rsplit("/", 1)[1]])
    properties = mapping(body["properties"])
    assert set(properties) == {
        "file", "model", "language", "response_format", "output_format", "with_timestamp",
    }
    assert body["required"] == ["file"]
    for name, default in (
        ("model", None), ("language", "auto"), ("response_format", None),
        ("output_format", "json"), ("with_timestamp", False),
    ):
        assert mapping(properties[name]).get("default") == default
    assert "default" not in mapping(properties["file"])
    for name, instructions in (
        ("file", ("200 MiB", "lower configured", "wav/mp3/m4a/mp4/flac/ogg/webm",
                  "externally", "costs credits", "no audio-minute cap")),
        ("model", ("deepgram-nova-3-diarize", "DEEPGRAM_ENABLED=true",
                   "DEEPGRAM_API_KEY", "never enter the key", "503", "resident local model")),
        ("language", ("'en' or 'zh'", "default 'auto'", "invalid", "400")),
        ("response_format", ("Deepgram", "verbose_json")),
    ):
        description = mapping(properties[name])["description"]
        assert isinstance(description, str)
        for instruction in instructions:
            assert instruction in description
    assert "DEEPGRAM_API_KEY" not in properties
