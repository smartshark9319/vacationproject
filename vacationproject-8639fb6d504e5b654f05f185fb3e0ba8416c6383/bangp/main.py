# app.py
from __future__ import annotations

import re
from typing import List

from fastapi import FastAPI, HTTPException, Body, Form
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

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
def _normalize_text(text: str) -> str:
    """
    입력 전처리:
    - 윈도우 개행(\r\n) → \n
    - 탭/여러 공백을 한 칸으로
    - 빈 줄(연속 개행)은 한 칸으로 축약
    """
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # 줄바꿈 주변 공백 제거 후 한 칸으로 축약
    t = re.sub(r"\s*\n\s*", " ", t)
    # 탭/다중 공백 축약
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()

def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    """
    긴 글을 대략 문장 경계 기준으로 분할.
    (정확한 토큰 수는 아님. 안전 운용 목적)
    """
    text = _normalize_text(text)
    # 문장 단위 분리(마침표/물음표/말줄임 뒤 공백)
    sentences = re.split(r"(?<=[.!?…])\s+", text) if text else []
    if not sentences:
        return [text] if text else []

    chunks: List[str] = []
    cur = ""
    for sent in sentences:
        # 첫 추가 시 앞 공백 금지
        candidate = (cur + (" " if cur else "") + sent)
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
    return chunks

def summarize_kor(text: str) -> List[str]:
    """기사 본문을 3줄로 요약"""
    chunks = chunk_text(text)
    if not chunks:
        return []

    chunk_summaries = []
    for ch in chunks:
        out = SUMMARIZER(ch, max_length=80, min_length=30, do_sample=False)
        chunk_summaries.append(out[0]["summary_text"])

    full_summary = " ".join(chunk_summaries).strip()
    # 3줄로 나누기
    lines = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", full_summary) if s.strip()]
    return lines[:3] if lines else [full_summary]

# ──────────────────────────────────────────────────────────────────────
# FastAPI 설정 및 API 엔드포인트
# ──────────────────────────────────────────────────────────────────────
app = FastAPI()

class SummarizeRequest(BaseModel):
    text: str  # 뉴스 본문

class SummarizeResponse(BaseModel):
    summary_3lines: List[str]  # 3줄 요약

# (A) JSON 바디: {"text": "여러 줄...\n그대로"}
@app.post("/summarize", response_model=SummarizeResponse)
def summarize_json(req: SummarizeRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="뉴스 본문이 필요합니다.")
    return SummarizeResponse(summary_3lines=summarize_kor(req.text))

# (B) text/plain 바디: 원문만 그대로 보냄
@app.post("/summarize-plain", response_model=SummarizeResponse)
def summarize_plain(text: str = Body(..., media_type="text/plain")):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="뉴스 본문이 필요합니다.")
    return SummarizeResponse(summary_3lines=summarize_kor(text))

# (C) form 바디: textarea로 전송 (application/x-www-form-urlencoded or multipart/form-data)
@app.post("/summarize-form", response_model=SummarizeResponse)
def summarize_form(text: str = Form(...)):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="뉴스 본문이 필요합니다.")
    return SummarizeResponse(summary_3lines=summarize_kor(text))

# ──────────────────────────────────────────────────────────────────────
# 서버 실행용 코드
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
