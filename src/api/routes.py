from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from typing import Optional, List
from pydantic import BaseModel, Field
import logging

# === 1. 定义响应模型 (The Contract) ===
# 这里就是你找的 "OpenAPI 定义"。
# 我们用 Python 类来代替 YAML，FastAPI 会自动把它们转换成文档。

class Segment(BaseModel):
    """对应 OpenAPI 中的 Segment 定义"""
    id: int = Field(description="片段ID")
    start: float = Field(description="开始时间(秒)")
    end: float = Field(description="结束时间(秒)")
    text: str = Field(description="片段文本")

class TranscriptionResponse(BaseModel):
    """
    对应 OpenAPI 中的 TranscriptionResponse 定义
    这是返回给客户端的最终结构
    """
    text: str = Field(description="转录文本（根据clean_tags参数决定是否清理）")
    task: str = Field(default="transcribe", description="任务类型")
    language: str = Field(default="zh", description="识别出的语言")
    duration: float = Field(description="音频总时长(秒)")
    model: Optional[str] = Field(default=None, description="实际使用的模型ID")
    raw_text: Optional[str] = Field(default=None, description="原始转录文本（包含所有SenseVoice标签和语气词）")
    is_cleaned: bool = Field(default=True, description="text字段是否经过清理")
    # 这里就是你觉得缺失的复杂部分：
    segments: Optional[List[Segment]] = Field(default=None, description="详细的时间戳分段信息")

# === 2. 路由定义 ===
router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/v1/audio/transcriptions",
    response_model=TranscriptionResponse,  # <--- 告诉 FastAPI：请按这个“模具”生成文档和校验返回值
    summary="语音转录接口",
    description="上传音频文件，返回转录文本。兼容 OpenAI Whisper 协议。",
    tags=["Audio"]
)
async def create_transcription(
    request: Request,
    # === 输入参数定义 (Request Body) ===
    # 这里的每一个参数，FastAPI 都会自动生成到 OpenAPI 的 requestBody 中
    file: UploadFile = File(..., description="音频文件 (wav, mp3, m4a)"),
    model: str = Form(default="auto", description="模型ID (信息用途，实际模型由启动配置决定)"),
    language: str = Form(default="auto", description="语言代码 (zh, en, ja, ko, auto)"),
    response_format: str = Form(default="json", description="返回格式 (json, verbose_json)"),
    clean_tags: bool = Form(default=True, description="是否清洗情感标签 (<happy>等)"),
    prompt: Optional[str] = Form(default=None, description="提示词 (当前版本未实装)"),
    temperature: float = Form(default=0.0, description="采样温度 (当前版本未实装)"),
):
    # 1. 获取 Service
    service = request.app.state.service

    try:
        # 2. 构造参数
        params = {
            "language": language,
            "clean_tags": clean_tags,
            "response_format": response_format
        }

        # 3. 提交任务 (Task Submission)
        # 这一步 result 拿到的其实是一个字典 (dict)
        result = await service.submit(file, params)
        
        # 4. 构造返回对象 (Data Mapping)
        # 如果 result 里没有 segments,Pydantic 会自动填 None，不会报错
        return TranscriptionResponse(
            text=result["text"],
            duration=result.get("duration", 0.0),
            language=language if language != "auto" else "zh", # MVP 简化处理
            model=request.app.state.model_id if hasattr(request.app.state, "model_id") else None,  # 返回实际使用的模型
            raw_text=result.get("raw_text"),  # 原始文本（包含所有标签）
            is_cleaned=result.get("is_cleaned", True),  # 是否清理过
            segments=result.get("segments", None) # 如果 Service 以后支持了 segments，这里直接透传
        )

    except RuntimeError as e:
        if "Queue is full" in str(e):
            raise HTTPException(status_code=503, detail="Server is busy (Queue Full). Please try again later.")
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        # 生产环境建议隐藏具体错误堆栈，但在 MVP 开发期打印出来方便调试
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")