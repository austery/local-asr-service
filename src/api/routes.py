"""
API routes for speech transcription service.
"""
from fastapi import APIRouter, File, UploadFile, Form, Request, HTTPException
from typing import Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

# API Request/Response Models
class Segment(BaseModel):
    """Segment with timestamp and optional speaker info"""
    id: int
    speaker: Optional[str] = None  # Speaker ID (e.g., "SPEAKER_00")
    start: float = 0.0
    end: float = 0.0
    text: str

class TranscriptionResponse(BaseModel):
    """
    OpenAI Whisper API compatible response format.
    Spec: https://platform.openai.com/docs/api-reference/audio/createTranscription
    """
    text: str
    duration: Optional[float] = None
    language: Optional[str] = None
    model: Optional[str] = None  # 返回实际使用的模型
    raw_text: Optional[str] = Field(None, description="转录前的原始文本（带所有标签）")
    is_cleaned: Optional[bool] = Field(True, description="是否经过清理")
    segments: Optional[list[Segment]] = Field(None, description="详细分段信息（带说话人识别）")

# Router
router = APIRouter()

@router.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    request: Request,
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, etc.)"),
    model: str = Form("sensevoice-small", description="Model ID (currently ignored, actual model in config)"),
    language: str = Form("auto", description="Language code (auto, zh, en, yue, ja, ko)"),
    response_format: str = Form("json", description="Response format: 'json' (text only) or 'verbose_json' (with speaker info)"),
    clean_tags: bool = Form(True, description="是否清理 <|xxx|> 标签（FunASR 专用）")
    ):
    # 1. 获取 Service
    service = request.app.state.service

    try:
        # 2. 确定是否需要说话人信息（verbose_json 或显式请求 segment）
        include_speaker_info = (response_format == "verbose_json")
        
        # 3. 构造参数
        params = {
            "language": language,
            "clean_tags": clean_tags,
            "response_format": "json" if include_speaker_info else "txt",  # MLX 引擎使用
        }

        # 4. 提交任务
        result = await service.submit(file, params)
        
        # 5. 处理返回值（可能是字符串或字典）
        if isinstance(result, dict):
            # MLX 引擎返回了 JSON 格式（包含说话人信息）
            text = result.get("text", "")
            segments_data = result.get("segments")
            duration = result.get("duration", 0.0)
            raw_text = result.get("raw_text")
            is_cleaned = result.get("is_cleaned", True)
            
            # 格式化 segments（添加 id 字段）
            segments = None
            if segments_data:
                segments = [
                    {
                        "id": i,
                        "speaker": seg.get("speaker"),
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "text": seg.get("text", "")
                    }
                    for i, seg in enumerate(segments_data)
                ]
        else:
            # 文本格式返回（FunASR 或 MLX txt 格式）
            text = result
            segments = None
            duration = 0.0
            raw_text = None
            is_cleaned = True
        
        # 6. 构造返回对象
        return TranscriptionResponse(
            text=text,
            duration=duration,
            language=language if language != "auto" else "zh",
            model=request.app.state.model_id if hasattr(request.app.state, "model_id") else None,
            raw_text=raw_text,
            is_cleaned=is_cleaned,
            segments=segments  # 说话人信息（如果有）
        )

    except RuntimeError as e:
        if "Queue is full" in str(e):
            raise HTTPException(status_code=503, detail="Server is busy (Queue Full). Please try again later.")
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
