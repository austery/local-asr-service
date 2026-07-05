from pathlib import Path

import pytest

from benchmarks.phase3_evaluation import (
    ProbeIdentity,
    ProbeOutcome,
    analyze_segments,
    build_probe_result,
    build_request_data,
    decode_json_object,
    parse_ffprobe_duration_stdout,
    parse_ps_process_table,
    parse_ps_rss_kb,
    process_tree_rss_mb_from_table,
    sample_process_rss_mb,
    summarize_json_response,
    summarize_srt_text,
)


def test_analyze_segments_distinguishes_null_empty_and_non_empty() -> None:
    assert analyze_segments({}).field_state == "missing"
    assert analyze_segments({"segments": None}).field_state == "null"
    empty = analyze_segments({"segments": []})
    assert empty.field_state == "empty"
    assert empty.monotonic is None

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


def test_analyze_segments_detects_cross_segment_ordering_defects() -> None:
    summary = analyze_segments(
        {
            "segments": [
                {"start": 1.0, "end": 2.0, "text": "first"},
                {"start": 1.5, "end": 3.0, "text": "overlap"},
            ]
        }
    )

    assert summary.field_state == "non_empty"
    assert summary.monotonic is False
    assert summary.zero_or_negative_duration_count == 0


def test_analyze_segments_keeps_zero_duration_separate_from_ordering() -> None:
    summary = analyze_segments(
        {
            "segments": [
                {"start": 0.0, "end": 0.0, "text": "zero"},
                {"start": 1.0, "end": 2.0, "text": "next"},
            ]
        }
    )

    assert summary.field_state == "non_empty"
    assert summary.monotonic is True
    assert summary.zero_or_negative_duration_count == 1


def test_analyze_segments_tracks_missing_timing_inside_non_empty_list() -> None:
    summary = analyze_segments(
        {"segments": [{"start": 0.0, "text": "missing end"}, {"start": 1.0, "end": 2.0}]}
    )

    assert summary.field_state == "non_empty"
    assert summary.count == 2
    assert summary.monotonic is False
    assert summary.missing_timing_count == 1


def test_analyze_segments_tracks_mixed_dict_and_non_dict_segments() -> None:
    summary = analyze_segments({"segments": [{"start": 0.0, "end": 1.0}, "bad"]})

    assert summary.field_state == "non_empty"
    assert summary.count == 1
    assert summary.monotonic is False
    assert summary.missing_timing_count == 1


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


def test_summarize_srt_text_reports_no_cues_as_not_applicable() -> None:
    summary = summarize_srt_text("plain text")

    assert summary.cue_count == 0
    assert summary.valid_format is False
    assert summary.monotonic is None


def test_summarize_srt_text_detects_non_monotonic_and_zero_duration_cues() -> None:
    summary = summarize_srt_text(
        "1\n00:00:01,000 --> 00:00:01,000\nzero\n\n"
        "2\n00:00:00,500 --> 00:00:02,000\noverlap\n"
    )

    assert summary.cue_count == 2
    assert summary.valid_format is True
    assert summary.monotonic is False


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


def test_process_tree_rss_sums_root_and_descendants_only() -> None:
    rows = parse_ps_process_table(
        """
          10     1   100
          11    10   200
          12    11   300
          13     1   400
        """
    )

    assert process_tree_rss_mb_from_table(10, rows) == round(600 / 1024, 3)


def test_process_tree_rss_returns_none_when_root_missing() -> None:
    rows = parse_ps_process_table("  11 10 200\n")

    assert process_tree_rss_mb_from_table(10, rows) is None


def test_decode_json_object_reports_invalid_json_and_non_object() -> None:
    payload, error = decode_json_object("not json")

    assert payload is None
    assert error is not None
    assert "valid JSON" in error

    payload, error = decode_json_object("[]")

    assert payload is None
    assert error == "JSON response body is not an object"


def test_parse_ffprobe_duration_stdout_rejects_blank_and_na_values() -> None:
    with pytest.raises(RuntimeError, match="empty duration"):
        parse_ffprobe_duration_stdout("", Path("sample.wav"))

    with pytest.raises(RuntimeError, match="N/A"):
        parse_ffprobe_duration_stdout("N/A\n", Path("sample.wav"))


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
