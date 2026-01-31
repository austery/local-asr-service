"""
音频切片服务 (Audio Chunking Service)

用于处理超长音频，提供智能切片功能。
核心策略：
1. 归一化：转换为 mono, 16kHz（Whisper 模型最优格式）
2. 智能切片：如果超过限制，在静音点切分（避免断词）

Reference: puresubs/AudioChunkingService.ts
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, NamedTuple
from dataclasses import dataclass

from src.config import (
    MAX_AUDIO_DURATION_MINUTES,
    SILENCE_THRESHOLD_SEC,
    SILENCE_NOISE_DB,
    AUDIO_SAMPLE_RATE,
    AUDIO_BITRATE,
    CHUNK_OVERLAP_SECONDS,
)


class AudioNormalizationResult(NamedTuple):
    """音频归一化结果"""
    normalized_path: str
    file_size_bytes: int
    duration_seconds: float


@dataclass
class SilenceInterval:
    """静音区间"""
    start: float  # 开始时间(秒)
    end: float    # 结束时间(秒)
    duration: float  # 持续时间(秒)


class AudioChunkingService:
    """
    音频切片服务
    
    支持两种切片策略：
    - 策略A (优先)：基于静音检测的智能切片
    - 策略B (fallback)：固定时长 + 重叠切片
    """
    
    def __init__(
        self,
        max_duration_minutes: int = MAX_AUDIO_DURATION_MINUTES,
        silence_threshold_sec: float = SILENCE_THRESHOLD_SEC,
        silence_noise_db: str = SILENCE_NOISE_DB,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        bitrate: str = AUDIO_BITRATE,
        overlap_seconds: int = CHUNK_OVERLAP_SECONDS,
    ):
        self.max_duration_seconds = max_duration_minutes * 60
        self.silence_threshold_sec = silence_threshold_sec
        self.silence_noise_db = silence_noise_db
        self.sample_rate = sample_rate
        self.bitrate = bitrate
        self.overlap_seconds = overlap_seconds
        
        # 检查 ffmpeg 和 ffprobe 是否可用
        self._check_ffmpeg_availability()
        
        print(f"🎛️  AudioChunkingService initialized:")
        print(f"   - Max duration: {max_duration_minutes} min")
        print(f"   - Silence threshold: {silence_threshold_sec}s @ {silence_noise_db}")
        print(f"   - Sample rate: {sample_rate}Hz, Bitrate: {bitrate}")
        print(f"   - Overlap duration: {overlap_seconds}s (fallback)")
    
    def _check_ffmpeg_availability(self):
        """检查 ffmpeg 和 ffprobe 是否可用"""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            subprocess.run(
                ["ffprobe", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "❌ ffmpeg and ffprobe are required for audio chunking. "
                "Install with: brew install ffmpeg"
            ) from e
    
    async def process_audio(self, input_path: str) -> List[str]:
        """
        处理音频：归一化并在必要时切片
        
        Args:
            input_path: 原始音频文件路径
            
        Returns:
            音频文件路径列表（如果不需要切片则只有一个路径）
        """
        print(f"🎵 Processing audio: {Path(input_path).name}")
        
        # Step 1: 归一化音频（mono, 16kHz）
        normalized = await self._normalize_audio(input_path)
        print(f"   ✓ Normalized: {normalized.file_size_bytes / 1024 / 1024:.2f}MB, "
              f"{normalized.duration_seconds:.1f}s")
        
        # Step 2: 检查是否需要切片
        if normalized.duration_seconds <= self.max_duration_seconds:
            print(f"   ✓ Duration OK, no chunking needed")
            return [normalized.normalized_path]
        
        # Step 3: 需要切片
        print(f"   ⚠️  Audio duration ({normalized.duration_seconds / 60:.1f} min) "
              f"exceeds limit ({self.max_duration_seconds / 60:.1f} min)")
        
        # 策略A: 尝试静音切片（自适应阈值）
        thresholds = ["-40dB", "-35dB", "-30dB", "-25dB"]
        for threshold in thresholds:
            print(f"   🔍 Trying silence-based splitting at {threshold}...")
            try:
                chunks = await self._try_silence_split(
                    normalized.normalized_path,
                    normalized.duration_seconds,
                    threshold,
                )
                if chunks:
                    print(f"   ✅ Success with silence splitting at {threshold}")
                    return chunks
            except Exception as e:
                print(f"   ⚠️  Failed at {threshold}: {e}")
                continue
        
        # 策略B: Fallback 到重叠切片
        print(f"   ⚠️  All silence detection attempts failed. Using overlap splitting.")
        return await self._split_with_overlap(
            normalized.normalized_path,
            normalized.duration_seconds,
        )
    
    async def _normalize_audio(self, input_path: str) -> AudioNormalizationResult:
        """
        归一化音频到最优格式
        - 单声道（语音不需要立体声）
        - 16kHz 采样率（Whisper 标准）
        - 适度压缩码率
        """
        input_p = Path(input_path)
        output_path = str(input_p.with_suffix(".normalized.mp3"))
        
        print(f"   🔧 Normalizing audio...")
        
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-ac", "1",  # Mono
            "-ar", str(self.sample_rate),  # 16kHz
            "-b:a", self.bitrate,  # 64k
            "-y",  # Overwrite
            output_path,
        ]
        
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
            
            file_size = Path(output_path).stat().st_size
            duration = await self._get_audio_duration(output_path)
            
            return AudioNormalizationResult(
                normalized_path=output_path,
                file_size_bytes=file_size,
                duration_seconds=duration,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Audio normalization failed: {e.stderr}") from e
    
    async def _get_audio_duration(self, audio_path: str) -> float:
        """使用 ffprobe 获取音频时长"""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            raise RuntimeError(f"Failed to get audio duration: {e}") from e
    
    async def _detect_silence(
        self,
        audio_path: str,
        threshold: str,
    ) -> List[SilenceInterval]:
        """
        使用 ffmpeg silencedetect 检测静音区间
        
        Args:
            audio_path: 音频文件路径
            threshold: 噪音阈值（如 "-30dB"）
            
        Returns:
            静音区间列表
        """
        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-af", f"silencedetect=noise={threshold}:d={self.silence_threshold_sec}",
            "-f", "null",
            "-",
        ]
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            output = result.stdout + result.stderr
            
            # 解析 silencedetect 输出
            # Format: [silencedetect @ ...] silence_start: 12.345
            #         [silencedetect @ ...] silence_end: 13.456 | silence_duration: 1.111
            silences: List[SilenceInterval] = []
            current_start: Optional[float] = None
            
            for line in output.split("\n"):
                start_match = re.search(r"silence_start:\s+([\d.]+)", line)
                end_match = re.search(r"silence_end:\s+([\d.]+)", line)
                
                if start_match:
                    current_start = float(start_match.group(1))
                elif end_match and current_start is not None:
                    end = float(end_match.group(1))
                    silences.append(SilenceInterval(
                        start=current_start,
                        end=end,
                        duration=end - current_start,
                    ))
                    current_start = None
            
            return silences
        except Exception as e:
            print(f"   ❌ Silence detection failed: {e}")
            return []
    
    async def _try_silence_split(
        self,
        audio_path: str,
        duration_seconds: float,
        threshold: str,
    ) -> List[str]:
        """
        尝试在静音点切分音频
        
        如果切分点不足或切片仍过大，返回空列表
        """
        # 1. 检测静音
        silences = await self._detect_silence(audio_path, threshold)
        if not silences:
            return []
        
        # 2. 计算理想切分点数量
        num_chunks = int((duration_seconds / self.max_duration_seconds) + 0.5)
        if num_chunks < 2:
            return []
        
        # 3. 计算理想切分时间点
        ideal_split_times = [
            (duration_seconds / num_chunks) * i
            for i in range(1, num_chunks)
        ]
        
        # 4. 将理想时间点对齐到最近的静音中点
        actual_split_times = [
            self._find_nearest_silence_midpoint(silences, ideal_time)
            for ideal_time in ideal_split_times
        ]
        
        # 5. 去重并验证
        unique_splits = sorted(set(actual_split_times))
        unique_splits = [
            t for t in unique_splits
            if 1.0 < t < duration_seconds - 1.0
        ]
        
        if not unique_splits and num_chunks > 1:
            return []
        
        # 6. 执行切分
        chunks = await self._split_audio_at_points(audio_path, unique_splits)
        
        # 7. 验证所有切片都在时长限制内
        all_valid = all(
            await self._get_audio_duration(chunk) <= self.max_duration_seconds
            for chunk in chunks
        )
        
        if not all_valid:
            print(f"      ❌ Some chunks still exceed limit. Discarding...")
            for chunk in chunks:
                Path(chunk).unlink(missing_ok=True)
            return []
        
        return chunks
    
    def _find_nearest_silence_midpoint(
        self,
        silences: List[SilenceInterval],
        target_time: float,
    ) -> float:
        """找到最接近目标时间的静音区间中点"""
        if not silences:
            return target_time
        
        def midpoint(s: SilenceInterval) -> float:
            return (s.start + s.end) / 2
        
        nearest = min(silences, key=lambda s: abs(midpoint(s) - target_time))
        return midpoint(nearest)
    
    async def _split_audio_at_points(
        self,
        audio_path: str,
        split_times: List[float],
    ) -> List[str]:
        """在指定时间点切分音频"""
        audio_p = Path(audio_path)
        duration = await self._get_audio_duration(audio_path)
        
        # 添加起点和终点
        all_points = [0.0] + split_times + [duration]
        
        chunk_paths: List[str] = []
        
        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i + 1]
            chunk_path = str(audio_p.with_suffix(f".chunk_{i}{audio_p.suffix}"))
            
            print(f"   ✂️  Creating chunk {i}: {start:.1f}s - {end:.1f}s")
            
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-ss", str(start),
                "-to", str(end),
                "-c", "copy",  # 不重新编码（快速）
                "-y",
                chunk_path,
            ]
            
            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                )
                chunk_paths.append(chunk_path)
                
                chunk_size = Path(chunk_path).stat().st_size
                print(f"      ✓ Chunk {i}: {chunk_size / 1024 / 1024:.2f}MB")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to create chunk {i}: {e.stderr}") from e
        
        return chunk_paths
    
    async def _split_with_overlap(
        self,
        audio_path: str,
        duration_seconds: float,
    ) -> List[str]:
        """
        使用固定时长 + 重叠策略切分音频
        
        当静音检测失败时使用此策略
        """
        # 计算切片数量（留 10% 安全裕度）
        num_chunks = int((duration_seconds / self.max_duration_seconds) + 1)
        chunk_duration = duration_seconds / num_chunks
        
        print(f"   ✂️  Overlap splitting: {num_chunks} chunks, "
              f"base duration ~{chunk_duration:.1f}s, "
              f"overlap {self.overlap_seconds}s")
        
        audio_p = Path(audio_path)
        chunk_paths: List[str] = []
        
        start_time = 0.0
        chunk_index = 0
        
        while start_time < duration_seconds:
            end_time = min(start_time + chunk_duration, duration_seconds)
            chunk_path = str(audio_p.with_suffix(f".chunk_ov_{chunk_index}{audio_p.suffix}"))
            
            print(f"      Generating chunk {chunk_index}: {start_time:.1f}s - {end_time:.1f}s")
            
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-c", "copy",
                "-y",
                chunk_path,
            ]
            
            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                )
                chunk_paths.append(chunk_path)
                
                chunk_size = Path(chunk_path).stat().st_size
                print(f"      ✓ Chunk {chunk_index}: {chunk_size / 1024 / 1024:.2f}MB")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to create overlap chunk {chunk_index}: {e.stderr}"
                ) from e
            
            if end_time >= duration_seconds:
                break
            
            # 向前推进，但回退重叠时长
            start_time = end_time - self.overlap_seconds
            chunk_index += 1
        
        return chunk_paths
