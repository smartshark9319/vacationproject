import re
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# ──────────────────────────────────────────────────────────────────────
# 모델 로드(요약)
# ──────────────────────────────────────────────────────────────────────
SUMMARIZER = pipeline(
    "summarization",
    model="gogamza/kobart-summarization",
    tokenizer="gogamza/kobart-summarization",
    device=-1  # CPU용
)

# ──────────────────────────────────────────────────────────────────────
# 유틸 함수들
# ──────────────────────────────────────────────────────────────────────
def chunk_text(text: str, max_tokens: int = 700) -> List[str]:
    """긴 글을 청크 단위로 분리"""
    sentences = re.split(r"(?<=[.!?…])\s+", text)  # 문장으로 분리
    chunks, current_chunk = [], ""
    
    # 각 문장을 chunk에 추가하여 토큰 수가 max_tokens를 넘지 않게 처리
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_tokens:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            
    if current_chunk:
        chunks.append(current_chunk.strip())  # 마지막 남은 부분 추가
    
    return chunks

def summarize_kor(text: str) -> List[str]:
    """기사 본문을 3줄로 요약"""
    # 긴 텍스트를 청크로 나누기
    chunks = chunk_text(text)
    
    # 각 청크를 요약
    chunk_summaries = [SUMMARIZER(chunk, max_length=80, min_length=30, do_sample=False)[0]["summary_text"] for chunk in chunks]
    
    # 각 청크 요약을 합치기
    full_summary = " ".join(chunk_summaries)
    
    # 3줄로 나누기
    summary_lines = re.split(r"(?<=[.!?…])\s+", full_summary)[:3]
    
    return summary_lines

# ──────────────────────────────────────────────────────────────────────
# FastAPI 설정 및 API 엔드포인트
# ──────────────────────────────────────────────────────────────────────
app = FastAPI()

class SummarizeRequest(BaseModel):
    text: str  # 뉴스 본문

class SummarizeResponse(BaseModel):
    summary_3lines: List[str]  # 3줄 요약

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="뉴스 본문이 필요합니다.")

    # 1. 뉴스 기사 요약
    summary_3lines = summarize_kor(req.text)

    # 응답 반환
    return SummarizeResponse(summary_3lines=summary_3lines)

# ──────────────────────────────────────────────────────────────────────
# 서버 실행용 코드
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
