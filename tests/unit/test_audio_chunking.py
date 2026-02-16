"""
测试 src/adapters/audio_chunking.py

策略：Mock 掉所有 subprocess.run 调用，避免依赖真实 ffmpeg。
重点验证：
- 归一化逻辑（跳过已优化的 WAV / 正常转码）
- 短音频直通（无切片）
- 静音检测解析
- 切分点对齐算法
- Fallback 重叠切片
"""
import struct
import wave
import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path

from src.adapters.audio_chunking import (
    AudioChunkingService,
    AudioNormalizationResult,
    SilenceInterval,
)


@pytest.fixture
def mock_ffmpeg():
    """Mock 所有 subprocess.run 调用"""
    with patch("src.adapters.audio_chunking.subprocess.run") as mock:
        # 默认让 ffmpeg/ffprobe 版本检查通过
        mock.return_value = MagicMock(
            stdout="", stderr="", returncode=0
        )
        yield mock


@pytest.fixture
def service(mock_ffmpeg):
    """创建 AudioChunkingService，跳过 ffmpeg 可用性检查"""
    return AudioChunkingService(
        max_duration_minutes=50,
        silence_threshold_sec=0.5,
        silence_noise_db="-30dB",
        sample_rate=16000,
        bitrate="64k",
        overlap_seconds=15,
    )


class TestNormalization:
    """测试音频归一化逻辑"""

    def test_skip_normalization_for_optimal_wav(self, service, mock_ffmpeg, tmp_path):
        """16kHz mono WAV 应该跳过 FFmpeg 转码 (uses Python wave module, no subprocess)"""
        # Create a real 16kHz mono WAV file (0.1s of silence)
        wav_file = tmp_path / "test.wav"
        num_frames = 1600  # 0.1s at 16kHz
        with wave.open(str(wav_file), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * num_frames)

        result = service._normalize_audio(str(wav_file))

        assert result.normalized_path == str(wav_file)
        assert abs(result.duration_seconds - 0.1) < 0.01
        # No ffprobe or ffmpeg calls for format detection — wave module handles it
        for c in mock_ffmpeg.call_args_list:
            assert c[0][0][0] != "ffmpeg" or "-version" in c[0][0]
            assert c[0][0][0] != "ffprobe" or "-version" in c[0][0]

    def test_normalize_non_wav(self, service, mock_ffmpeg, tmp_path):
        """非 WAV 文件应该执行 FFmpeg 转码"""
        mp3_file = tmp_path / "test.mp3"
        mp3_file.write_bytes(b"\x00" * 100)

        # 需要创建输出文件让 Path.stat() 不报错
        output_path = tmp_path / "test.normalized.wav"
        output_path.write_bytes(b"\x00" * 50)

        def side_effect(*args, **kwargs):
            cmd = args[0]
            result = MagicMock()
            result.stdout = ""
            result.stderr = ""
            result.returncode = 0
            if cmd[0] == "ffprobe" and "format=duration" in cmd:
                result.stdout = "25.0"
            return result

        mock_ffmpeg.side_effect = side_effect

        result = service._normalize_audio(str(mp3_file))

        assert result.normalized_path == str(output_path)
        assert result.duration_seconds == 25.0


class TestProcessAudio:
    """测试 process_audio 主流程"""

    def test_short_audio_no_chunking(self, service, mock_ffmpeg, tmp_path):
        """短音频（<50min）应直接返回，不切片"""
        # Create a real 16kHz mono WAV file (2 minutes = 120s)
        wav_file = tmp_path / "short.wav"
        num_frames = 16000 * 120  # 120s at 16kHz
        with wave.open(str(wav_file), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * num_frames)

        chunks = service.process_audio(str(wav_file))

        assert len(chunks) == 1
        assert chunks[0] == str(wav_file)


class TestSilenceDetection:
    """测试静音检测解析"""

    def test_parse_silence_output(self, service, mock_ffmpeg):
        """测试 ffmpeg silencedetect 输出解析"""
        # 模拟 ffmpeg 的 silencedetect 输出
        ffmpeg_output = (
            "[silencedetect @ 0x1234] silence_start: 10.5\n"
            "[silencedetect @ 0x1234] silence_end: 11.2 | silence_duration: 0.7\n"
            "[silencedetect @ 0x1234] silence_start: 25.0\n"
            "[silencedetect @ 0x1234] silence_end: 26.5 | silence_duration: 1.5\n"
        )

        mock_ffmpeg.return_value = MagicMock(
            stdout="",
            stderr=ffmpeg_output,
            returncode=0,
        )

        silences = service._detect_silence("test.wav", "-30dB")

        assert len(silences) == 2
        assert silences[0].start == 10.5
        assert silences[0].end == 11.2
        assert abs(silences[0].duration - 0.7) < 0.01
        assert silences[1].start == 25.0
        assert silences[1].end == 26.5

    def test_no_silence_found(self, service, mock_ffmpeg):
        """没有检测到静音时返回空列表"""
        mock_ffmpeg.return_value = MagicMock(
            stdout="",
            stderr="some other ffmpeg output\n",
            returncode=0,
        )

        silences = service._detect_silence("test.wav", "-30dB")

        assert silences == []

    def test_incomplete_silence_pair(self, service, mock_ffmpeg):
        """只有 silence_start 没有 silence_end 时忽略"""
        ffmpeg_output = (
            "[silencedetect @ 0x1234] silence_start: 10.5\n"
            # 没有对应的 silence_end
        )

        mock_ffmpeg.return_value = MagicMock(
            stdout="",
            stderr=ffmpeg_output,
            returncode=0,
        )

        silences = service._detect_silence("test.wav", "-30dB")

        assert silences == []


class TestSplitPointAlignment:
    """测试切分点对齐到静音中点的算法"""

    def test_find_nearest_silence_midpoint(self, service):
        """理想切分时间应该对齐到最近的静音中点"""
        silences = [
            SilenceInterval(start=10.0, end=12.0, duration=2.0),  # midpoint=11.0
            SilenceInterval(start=25.0, end=27.0, duration=2.0),  # midpoint=26.0
            SilenceInterval(start=55.0, end=57.0, duration=2.0),  # midpoint=56.0
        ]

        # 目标时间 30.0 应该对齐到 midpoint=26.0（最近）
        result = service._find_nearest_silence_midpoint(silences, 30.0)
        assert result == 26.0

        # 目标时间 10.0 应该对齐到 midpoint=11.0（最近）
        result = service._find_nearest_silence_midpoint(silences, 10.0)
        assert result == 11.0

    def test_empty_silences_returns_target(self, service):
        """没有静音区间时返回原始目标时间"""
        result = service._find_nearest_silence_midpoint([], 30.0)
        assert result == 30.0


class TestSRTTimestamp:
    """测试 SRT 时间格式转换（FunASR 引擎中的辅助方法）"""

    def test_ms_to_srt_time(self):
        """验证毫秒到 SRT 时间格式的转换"""
        from src.core.funasr_engine import FunASREngine

        engine = FunASREngine.__new__(FunASREngine)

        assert engine._ms_to_srt_time(0) == "00:00:00,000"
        assert engine._ms_to_srt_time(5000) == "00:00:05,000"
        assert engine._ms_to_srt_time(65000) == "00:01:05,000"
        assert engine._ms_to_srt_time(3661500) == "01:01:01,500"
        assert engine._ms_to_srt_time(-100) == "00:00:00,000"  # 负值保护


class TestSRTFormat:
    """测试 FunASR 引擎的 SRT 格式输出"""

    def test_format_as_srt(self):
        """验证 SRT 格式输出正确"""
        from src.core.funasr_engine import FunASREngine

        engine = FunASREngine.__new__(FunASREngine)

        sentence_info = [
            {"text": "Hello World", "start": 5000, "end": 20000, "spk": 0},
            {"text": "How are you", "start": 21000, "end": 35000, "spk": 1},
        ]

        result = engine._format_as_srt(sentence_info)

        assert "1\n" in result
        assert "00:00:05,000 --> 00:00:20,000" in result
        assert "[Speaker 0]: Hello World" in result
        assert "2\n" in result
        assert "00:00:21,000 --> 00:00:35,000" in result
        assert "[Speaker 1]: How are you" in result
