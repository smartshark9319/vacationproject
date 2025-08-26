import re
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import feedparser
import requests
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────────────────────────────
# 모델 로드(요약) - kobart
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "gogamza/kobart-summarization"
SUMMARIZER = pipeline(
    task="summarization",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=-1  # CPU
)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_INPUT_TOKENS = 900
CHUNK_OVERLAP = 100

# ──────────────────────────────────────────────────────────────────────
# 유틸 함수들
# ──────────────────────────────────────────────────────────────────────
def sentence_split_ko(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r"(?<=[\.!?…])\s+", text)

def tokenize_len(txt: str) -> int:
    return len(TOKENIZER.encode(txt, add_special_tokens=False))

def chunk_by_tokens(text: str,
                    max_tokens: int = MAX_INPUT_TOKENS,
                    overlap: int = CHUNK_OVERLAP) -> List[str]:
    sents = sentence_split_ko(text)
    if not sents:
        return []
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0

    i = 0
    while i < len(sents):
        sent = sents[i]
        stoks = tokenize_len(sent)

        if cur_tokens + stoks <= max_tokens:
            cur.append(sent)
            cur_tokens += stoks
            i += 1
        else:
            if cur:
                chunks.append(" ".join(cur).strip())
            else:
                hard = sent[:2000]
                chunks.append(hard)
                sents[i] = sent[2000:]
                continue

            if overlap > 0:
                overlap_sents: List[str] = []
                overlap_tok = 0
                for s in reversed(cur):
                    t = tokenize_len(s)
                    if overlap_tok + t <= overlap:
                        overlap_sents.insert(0, s)
                        overlap_tok += t
                    else:
                        break
                cur = overlap_sents[:]
                cur_tokens = sum(tokenize_len(s) for s in cur)
            else:
                cur = []
                cur_tokens = 0

    if cur:
        chunks.append(" ".join(cur).strip())

    chunks = [c for c in chunks if len(c) > 20]
    return chunks

def summarize_chunk(chunk: str,
                    max_summary_len: int = 110,
                    min_summary_len: int = 40) -> str:
    out = SUMMARIZER(
        chunk,
        max_length=max_summary_len,
        min_length=min_summary_len,
        do_sample=False
    )
    return out[0]["summary_text"].strip()

def compress_to_3_lines(text: str) -> List[str]:
    sents = sentence_split_ko(text)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return []
    return sents[:3]

def summarize_kor(text: str) -> List[str]:
    chunks = chunk_by_tokens(text, MAX_INPUT_TOKENS, CHUNK_OVERLAP)
    if not chunks:
        return []
    partials = [summarize_chunk(c) for c in chunks]
    merged = " ".join(partials)
    three = compress_to_3_lines(merged)
    if len(three) < 3 and len(partials) > 1:
        extra = sentence_split_ko(" ".join(partials[1:]))
        for s in extra:
            if len(three) >= 3:
                break
            if s not in three:
                three.append(s)
    return three[:3]

def get_latest_news_url() -> str | None:
    rss_url = "https://news.google.com/rss/search?q=site:naver.com&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(rss_url)
    for entry in feed.entries:
        return entry.link
    return None

def get_news_content_naver(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if not res.encoding or res.encoding.lower() == "iso-8859-1":
            res.encoding = res.apparent_encoding or "utf-8"
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"요청 실패: {e}")

    selectors = [
        "#dic_area",
        ".newsct_article",
        "#articeBody",
        "#articleBodyContents",
    ]
    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            txt = node.get_text(separator="\n", strip=True)
            lines = [ln.strip() for ln in txt.splitlines() if len(ln.strip()) >= 2]
            return "\n".join(lines)

    return get_visible_text(url)

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    words = re.findall(r"\b[가-힣]{2,}\b", text)
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]

def get_visible_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if not res.encoding or res.encoding.lower() == "iso-8859-1":
            res.encoding = res.apparent_encoding or "utf-8"
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"요청 실패: {e}")

    soup = BeautifulSoup(res.text, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form",
                     "noscript", "iframe", "button"]):
        tag.decompose()
    for ad_tag in soup.find_all(attrs={"class": re.compile("ad|banner", re.I)}):
        ad_tag.decompose()
    for ad_tag in soup.find_all(attrs={"id": re.compile("ad|banner", re.I)}):
        ad_tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    lines = [line for line in text.splitlines() if len(line.strip()) > 20]
    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────
# FastAPI 설정 및 API 엔드포인트
# ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="KoBART News Summarizer", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def root():
    return {
        "message": "KoBART summarizer is running.",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/summarize", "/summarize-url", "/latest-naver"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

class SummarizeRequest(BaseModel):
    text: str

class SummarizeResponse(BaseModel):
    summary_3lines: List[str]

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="뉴스 본문이 필요합니다.")
    summary_3lines = summarize_kor(text)
    if not summary_3lines:
        raise HTTPException(status_code=500, detail="요약에 실패했습니다.")
    return SummarizeResponse(summary_3lines=summary_3lines)

class UrlIn(BaseModel):
    url: str

class UrlOut(BaseModel):
    summary_3lines: List[str]
    keywords: List[str]
    used_url: str

@app.post("/summarize-url", response_model=UrlOut)
def summarize_url(data: UrlIn):
    url = (data.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL이 필요합니다.")
    content = get_news_content_naver(url)
    if not content or len(content) < 80:
        content = get_visible_text(url)
    if not content or len(content) < 80:
        raise HTTPException(status_code=404, detail="본문을 찾을 수 없습니다.")

    summary_3lines = summarize_kor(content)
    if not summary_3lines:
        raise HTTPException(status_code=500, detail="요약에 실패했습니다.")
    keywords = extract_keywords(content)
    return UrlOut(summary_3lines=summary_3lines, keywords=keywords, used_url=url)

class LatestOut(BaseModel):
    latest_url: str
    summary_3lines: List[str]
    keywords: List[str]

@app.get("/latest-naver", response_model=LatestOut)
def latest_naver():
    url = get_latest_news_url()
    if not url:
        raise HTTPException(status_code=404, detail="최신 뉴스 URL을 찾을 수 없습니다.")
    content = get_news_content_naver(url)
    if not content or len(content) < 80:
        content = get_visible_text(url)
    if not content or len(content) < 80:
        raise HTTPException(status_code=404, detail="최신 뉴스 본문을 찾을 수 없습니다.")
    summary_3lines = summarize_kor(content)
    if not summary_3lines:
        raise HTTPException(status_code=500, detail="요약에 실패했습니다.")
    keywords = extract_keywords(content)
    return LatestOut(latest_url=url, summary_3lines=summary_3lines, keywords=keywords)

# ──────────────────────────────────────────────────────────────────────
# 서버 실행
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # ★ 포트 충돌/권한 문제 회피: 8765 사용
    uvicorn.run(app, host="127.0.0.1", port=8765)
