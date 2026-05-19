#!/usr/bin/env python3
"""
SPEC-007 Speaker Diarization Demo Script

验证 FunASR + Cam++ 说话人分离功能和多格式输出。
使用 tests/fixtures/ 下的测试音频文件。

Usage:
    uv run python examples/demo_diarization.py
"""

import os
import sys
import time

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.funasr_engine import FunASREngine


def main():
    # 测试音频文件路径
    test_audio = "tests/fixtures/Paul Rosolie and Lex Fridman.mp3"

    if not os.path.exists(test_audio):
        print(f"❌ Test audio not found: {test_audio}")
        print("   Please copy test audio to tests/fixtures/")
        sys.exit(1)

    print("=" * 60)
    print("🎙️  SPEC-007 Speaker Diarization Demo")
    print("=" * 60)
    print(f"📁 Audio file: {test_audio}")
    print()

    # 1. 初始化引擎
    print("⏳ Initializing FunASR engine with Paraformer + Cam++...")
    engine = FunASREngine()

    # 2. 加载模型
    print("⏳ Loading models (this may take a while on first run)...")
    load_start = time.time()
    engine.load()
    load_time = time.time() - load_start
    print(f"✅ Models loaded in {load_time:.2f}s")
    print()

    # ============================================
    # 测试 1: JSON 格式 (完整数据)
    # ============================================
    print("=" * 60)
    print("📊 Test 1: JSON Format (Full Data)")
    print("=" * 60)

    infer_start = time.time()
    result_json = engine.transcribe_file(test_audio, output_format="json")
    infer_time = time.time() - infer_start
    print(f"✅ Transcription completed in {infer_time:.2f}s")

    segments = result_json.get("segments", [])
    if segments:
        print(f"✅ Found {len(segments)} segments")
        speakers = set(seg["speaker"] for seg in segments)
        print(f"👥 Detected speakers: {', '.join(sorted(speakers))}")
        print("\n📝 Sample (first 3 segments):")
        for seg in segments[:3]:
            print(f"  - {seg}")
    print()

    # ============================================
    # 测试 2: TXT 格式 (纯净文本，适合 RAG/LLM)
    # ============================================
    print("=" * 60)
    print("📝 Test 2: TXT Format (Clean Text for RAG/LLM)")
    print("=" * 60)

    result_txt = engine.transcribe_file(test_audio, output_format="txt", with_timestamp=False)
    print("📄 Clean text output (first 500 chars):")
    print("-" * 40)
    print(result_txt[:500])
    print("-" * 40)
    print()

    # ============================================
    # 测试 3: TXT 格式 (带时间戳)
    # ============================================
    print("=" * 60)
    print("⏱️  Test 3: TXT Format (With Timestamp)")
    print("=" * 60)

    result_txt_ts = engine.transcribe_file(test_audio, output_format="txt", with_timestamp=True)
    print("📄 Text with timestamps (first 500 chars):")
    print("-" * 40)
    print(result_txt_ts[:500])
    print("-" * 40)
    print()

    # ============================================
    # 测试 4: SRT 字幕格式
    # ============================================
    print("=" * 60)
    print("🎬 Test 4: SRT Subtitle Format")
    print("=" * 60)

    result_srt = engine.transcribe_file(test_audio, output_format="srt")
    print("📄 SRT output (first 800 chars):")
    print("-" * 40)
    print(result_srt[:800])
    print("-" * 40)
    print()

    # 完成
    print("=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)

    # 释放资源
    engine.release()


if __name__ == "__main__":
    main()

