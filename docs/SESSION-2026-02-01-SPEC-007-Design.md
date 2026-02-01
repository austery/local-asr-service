# 会话总结：SPEC-007 说话人分离设计与集成规划

**日期**: 2026-02-01  
**主题**: 集成 FunASR 原生说话人分离模型 (Cam++)  
**规范文档**: [SPEC-007-Speaker-Diarization-FunASR.md](./SPEC-007-Speaker-Diarization-FunASR.md)

---

## 📊 达成共识

本次会话确立了说话人分离的最终技术路线，放弃了复杂的 `pyannote` 方案，转向更加原生且轻量级的 `cam++` 集成方案。

### 核心决策
1. **模型组合**: 采用 `SenseVoiceSmall` + `fsmn-vad` + `cam++` 的阿里开源组合。
2. **集成方式**: 直接在 `FunASREngine` 的 `AutoModel` 中启用 `spk_model="cam++"`。
3. **输出格式**: 优先推荐 `List[Dict]` 结构化数据，以支持 EMR (电子病历) 系统的证据提取需求。

## ✅ 已完成工作 (Documentation Only)

1. **[NEW] [SPEC-007-Speaker-Diarization-FunASR.md](file:///Users/leipeng/Documents/Projects/local-asr-service/docs/SPEC-007-Speaker-Diarization-FunASR.md)**: 创建了详细的技术规范，包含模型加载和结果解析的伪代码。
2. **[UPDATE] [SPEC-005-Speaker-Diarization.md](file:///Users/leipeng/Documents/Projects/local-asr-service/docs/SPEC-005-Speaker-Diarization.md)**: 将方案 C (FunASR + Cam++) 标记为**已选中 (Chosen)**，并指向 SPEC-007。

## 🔧 实施路线图 (Next Steps)

### Phase 1: 核心引擎更新
- [ ] 修改 `src/core/funasr_engine.py` 的 `load` 方法。
- [ ] 修改 `src/core/funasr_engine.py` 的 `transcribe_file` 方法以支持 `sentence_info` 解析。

### Phase 2: API 与数据清洗
- [ ] 确保 `text_cleaner` 能够处理带说话人标记的输出（如果需要）。
- [ ] 更新 API 响应模型以包含 `speaker` 信息。

### Phase 3: 验证
- [ ] 在 `tests/e2e` 中添加包含多说话人的测试音频。
- [ ] 验证在 M4 Pro 上的首跳延迟和显存占用。

---

**会话记录**: 本次会话由用户提供核心技术改造代码，由 Antigravity 整理为正式 Specification 文档。
