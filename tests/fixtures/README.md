# Test Fixtures

此目录用于存放测试用的音频文件。

## ⚠️ 注意事项

音频文件（`.mp3`, `.wav`, `.m4a`）已被 `.gitignore` 排除，不会上传到 Git 仓库。
请在本地手动准备以下测试文件。

## 📁 测试文件

| 文件名 | 描述 | 用途 |
|-------|------|------|
| `two_speakers_60s.wav` | 60s English dialogue (16kHz mono WAV, ~1.8MB), 2 speakers (Lex Fridman + Peter Steinberger) | Speaker diarization test |
| `Blair_FEB_09_last10min.wav` | VP presentation final Q&A segment (~10 min WAV), multiple speakers with audience handoff | SPEC-012 realistic diarization probe |
| `Paul Rosolie and Lex Fridman.normalized.mp3` | Normalized long-form conversation sample | Secondary reference sample for SPEC-012 |

## 🔗 获取测试文件

如果你是新 clone 本项目的开发者，请联系项目维护者获取测试音频，
或自行准备类似特征的测试文件（多人对话，至少 2 个说话人）。

## Qwen automatic-language E2E

`tests/e2e/test_qwen_auto_language.py` requires the existing
`two_speakers_60s.wav` English dialogue and cached
`mlx-community/Qwen3-ASR-1.7B-8bit` weights on Apple Silicon. It creates a
task-local 21-minute WAV by prepending 20 minutes of digital silence and
checks that public `language=auto` retains the speech after the silent first
internal chunk. Use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to enforce cached-weight testing;
the fixture must already be available.
