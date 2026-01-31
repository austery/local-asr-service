"""
演示如何使用 clean_tags 参数控制文本清理
"""

import requests

# API 端点
url = "http://localhost:50070/v1/audio/transcriptions"

# 示例 1: 清理标签（默认行为）
print("=" * 60)
print("示例 1: clean_tags=True (默认) - 清理所有标签")
print("=" * 60)

with open("test_audio.wav", "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        data={
            "language": "zh",
            "clean_tags": "true"  # 清理标签
        }
    )
    result = response.json()
    print(f"清理后文本: {result['text']}")
    print(f"原始文本: {result['raw_text']}")
    print(f"是否清理: {result['is_cleaned']}")
    print()

# 示例 2: 保留原始标签
print("=" * 60)
print("示例 2: clean_tags=False - 保留所有原始标签和语气词")
print("=" * 60)

with open("test_audio.wav", "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        data={
            "language": "zh",
            "clean_tags": "false"  # 不清理，保留原始输出
        }
    )
    result = response.json()
    print(f"文本 (未清理): {result['text']}")
    print(f"原始文本: {result['raw_text']}")
    print(f"是否清理: {result['is_cleaned']}")
    print()

print("=" * 60)
print("说明:")
print("- text: 主要文本字段，根据 clean_tags 参数决定是否清理")
print("- raw_text: 始终包含原始模型输出（包含 <|zh|>、<|NEUTRAL|> 等标签）")
print("- is_cleaned: 标记 text 字段是否经过清理")
print("=" * 60)
