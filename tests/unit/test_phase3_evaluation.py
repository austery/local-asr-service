from benchmarks.phase3_evaluation import (
    ProbeIdentity,
    ProbeOutcome,
    analyze_segments,
    build_probe_result,
    build_request_data,
    parse_ps_rss_kb,
    sample_process_rss_mb,
    summarize_json_response,
    summarize_srt_text,
)


def test_analyze_segments_distinguishes_null_empty_and_non_empty() -> None:
    assert analyze_segments({}).field_state == "missing"
    assert analyze_segments({"segments": None}).field_state == "null"
    assert analyze_segments({"segments": []}).field_state == "empty"

    summary = analyze_segments({"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]})

    assert summary.field_state == "non_empty"
    assert summary.count == 1
    assert summary.monotonic is True


def test_analyze_segments_detects_timing_defects() -> None:
    summary = analyze_segments(
        {
            "segments": [
                {"start": 2.0, "end": 1.0, "text": "bad"},
                {"start": 0.5, "end": 0.25, "text": "also bad"},
            ]
        }
    )

    assert summary.field_state == "non_empty"
    assert summary.monotonic is False
    assert summary.zero_or_negative_duration_count == 2


def test_analyze_segments_marks_all_non_dict_items_as_invalid() -> None:
    summary = analyze_segments({"segments": ["string", 42, None]})

    assert summary.field_state == "invalid"
    assert summary.count == 0
    assert summary.monotonic is False
    assert summary.missing_timing_count == 3
    assert summary.zero_or_negative_duration_count == 0


def test_summarize_json_response_keeps_model_language_and_preview() -> None:
    summary = summarize_json_response(
        {
            "text": "hello world",
            "model": "apple-speech",
            "language": "en-US",
            "duration": 1.0,
            "segments": [],
        }
    )

    assert summary.model == "apple-speech"
    assert summary.language == "en-US"
    assert summary.text_length == 11
    assert summary.text_preview == "hello world"
    assert summary.segments.field_state == "empty"


def test_summarize_srt_text_detects_valid_monotonic_cues() -> None:
    summary = summarize_srt_text(
        "1\n00:00:00,000 --> 00:00:01,250\nhello\n\n"
        "2\n00:00:01,250 --> 00:00:02,000\nworld\n"
    )

    assert summary.cue_count == 2
    assert summary.valid_format is True
    assert summary.monotonic is True


def test_build_request_data_requires_explicit_language() -> None:
    assert build_request_data("apple-speech", "zh-CN", "json") == {
        "model": "apple-speech",
        "language": "zh-CN",
        "output_format": "json",
    }


def test_parse_ps_rss_kb_handles_blank_and_numbers() -> None:
    assert parse_ps_rss_kb("  204800\n") == 204800
    assert parse_ps_rss_kb("") is None


def test_parse_ps_rss_kb_rejects_non_numeric_output() -> None:
    assert parse_ps_rss_kb("RSS\n") is None


def test_build_probe_result_computes_runtime_ratios() -> None:
    result = build_probe_result(
        ProbeIdentity(
            model="apple-speech",
            language="en-US",
            file_name="sample.wav",
            audio_duration_s=10.0,
            file_size_mb=1.5,
        ),
        ProbeOutcome(
            elapsed_s=2.0,
            status_code=200,
            json_summary=summarize_json_response(
                {
                    "text": "hello world",
                    "model": "apple-speech",
                    "language": "en-US",
                    "segments": [],
                }
            ),
            srt_summary=None,
            peak_rss_mb=128.0,
            error=None,
        ),
    )

    assert result.rtf == 0.2
    assert result.speed_ratio == 5.0
    assert result.peak_rss_mb == 128.0
    assert result.json_summary is not None


def test_sample_process_rss_mb_returns_none_for_unavailable_process() -> None:
    assert sample_process_rss_mb(-1) is None
