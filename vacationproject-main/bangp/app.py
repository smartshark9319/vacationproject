# app.py
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, AnyHttpUrl
from typing import List, Dict
from bs4 import BeautifulSoup
import requests, re, json

# ─────────────────────────────────────────────────────────────────────
# 단일 FastAPI 앱
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="뉴스 요약 API", version="2.0.0")

# CORS (필요 시 도메인 제한 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적파일: 현재 폴더를 /static 경로로 서빙 (index.html 포함)
app.mount("/static", StaticFiles(directory="."), name="static")

# 모든 API는 /api 아래로 통일
api = APIRouter(prefix="/api", tags=["api"])

# ─────────────────────────────────────────────────────────────────────
# 저장소(옵션)
# ─────────────────────────────────────────────────────────────────────
SUMMARY_FILE = "summary.json"

def load_summary() -> str:
    try:
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("summary", "")
    except FileNotFoundError:
        return ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def save_summary(summary: str):
    try:
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────
# 모델 (요청/응답)
# ─────────────────────────────────────────────────────────────────────
class SummaryIn(BaseModel):
    summary: str

class AnswerCheckIn(BaseModel):
    user_answer: str
    correct_answer: str

class AnswerCheckOut(BaseModel):
    correct: bool
    similarity: float

class UrlIn(BaseModel):
    url: AnyHttpUrl  # http/https 유효성 자동 검증

class UrlSummaryOut(BaseModel):
    summary_3lines: List[str]
    keywords: List[str]
    meta: Dict[str, str]

# ─────────────────────────────────────────────────────────────────────
# 텍스트 유틸/요약 (가벼운 버전: 외부 모델 없이 동작)
# ─────────────────────────────────────────────────────────────────────
# Python 3.13 호환: 가변폭 룩비하인드 제거한 문장분리
SENT_SPLIT = re.compile(r"(?<=[\.!?…])\s+|(?<=[가-힣]\))\s+|\n+")
NON_TEXT   = re.compile(r"\s+")

def split_sentences(text: str) -> List[str]:
    text = NON_TEXT.sub(" ", text).strip()
    parts = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    # 너무 짧은(광고/버튼) 문장 제거
    return [s for s in parts if len(s) >= 8]

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    words = re.findall(r"[가-힣]{2,}", text)
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]

def summarize_3lines(text: str) -> List[str]:
    """
    (간단 버전) 빈도 기반 문장 점수 요약:
      1) 단어 빈도 딕셔너리
      2) 문장 점수 = 포함 단어 빈도 합
      3) 상위 3문장 정렬(원문 순서 보존)
    """
    sents = split_sentences(text)
    if not sents:
        return []
    words = re.findall(r"[가-힣]{2,}", text)
    if not words:
        # 한국어가 거의 없으면 앞 3개 반환
        return sents[:3]

    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    scored = []
    for s in sents:
        wlist = re.findall(r"[가-힣]{2,}", s)
        score = sum(freq.get(w, 0) for w in wlist)
        if len(s) < 20:  # 너무 짧은 제목/캡션류 패널티
            score *= 0.6
        scored.append((s, score))

    # 상위 3개(원문 순서 보존 위해 2단계 정렬)
    top = sorted(scored, key=lambda x: (-x[1], sents.index(x[0])))[:3]
    top_in_order = [t[0] for t in sorted(top, key=lambda x: sents.index(x[0]))]
    return top_in_order

# ─────────────────────────────────────────────────────────────────────
# 본문 크롤링
# ─────────────────────────────────────────────────────────────────────
REQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://news.naver.com/"
}

MAIN_SELECTORS = [
    "#dic_area",                 # 네이버 최신 본문
    ".newsct_article",           # 네이버 일부 템플릿
    "#articeBody", "#articleBodyContents",  # 구형 템플릿
    "article .content", "article .article_body", "article .article-body",
    ".article_body", ".article-body", "#newsEndContents",
    "div.newsct_article._article_content", "div#newsct_article"
]

REMOVE_TAGS = ["script", "style", "header", "footer", "nav", "aside", "form", "noscript", "iframe", "button"]
RM_CLASS = re.compile(r"(ad|banner|footer|header|promo|subscribe|comment|related)", re.I)
RM_ID    = re.compile(r"(ad|banner|footer|header|promo|subscribe|comment|related)", re.I)

def fetch_html(url: str) -> BeautifulSoup:
    try:
        res = requests.get(url, headers=REQ_HEADERS, timeout=10)
        res.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"요청 실패: {e}")
    return BeautifulSoup(res.text, "html.parser")

def extract_by_selectors(soup: BeautifulSoup) -> str:
    for sel in MAIN_SELECTORS:
        node = soup.select_one(sel)
        if node:
            txt = node.get_text(separator="\n", strip=True)
            if txt and len(txt) > 120:
                return txt
    return ""

def visible_fallback(soup: BeautifulSoup) -> str:
    # 노이즈 태그 제거
    for tag in soup(REMOVE_TAGS):
        tag.decompose()
    for t in soup.find_all(attrs={"class": RM_CLASS}):
        t.decompose()
    for t in soup.find_all(attrs={"id": RM_ID}):
        t.decompose()
    txt = soup.get_text(separator="\n", strip=True)
    lines = [ln for ln in txt.splitlines() if len(ln.strip()) > 20]
    return "\n".join(lines)

def get_article_text(url: str) -> str:
    soup = fetch_html(url)

    # 1) 지정 셀렉터 우선
    text = extract_by_selectors(soup)
    if text:
        return text

    # 2) AMP 태그 제거 후 재시도
    for tag in soup.find_all(re.compile(r"^amp-", re.I)):
        tag.decompose()
    soup2 = BeautifulSoup(soup.decode(), "html.parser")
    text = extract_by_selectors(soup2)
    if text:
        return text

    # 3) 가시 텍스트 백업
    fb = visible_fallback(soup2)
    if not fb or len(fb) < 120:
        raise HTTPException(status_code=404, detail="기사 본문을 찾을 수 없습니다.")
    return fb

# ─────────────────────────────────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "뉴스 요약 API 서버입니다.",
        "docs": "/docs",
        "static": "/static/index.html"
    }

@api.get("/health")
def health():
    return {"status": "ok"}

@api.get("/summary")
def get_summary_api():
    return {"summary": load_summary()}

@api.post("/summary")
def update_summary_api(data: SummaryIn):
    save_summary(data.summary)
    return {"message": "요약이 저장되었습니다.", "summary": data.summary}

@api.post("/check-answer", response_model=AnswerCheckOut)
def check_answer_api(data: AnswerCheckIn):
    def clean_text(t: str) -> str:
        return re.sub(r'[^가-힣a-zA-Z0-9\s]', '', t.lower())

    correct_clean = clean_text(data.correct_answer)
    user_clean = clean_text(data.user_answer)
    correct_words = set(correct_clean.split())
    user_words = set(user_clean.split())

    if not correct_words:
        return AnswerCheckOut(correct=False, similarity=0.0)

    overlap = correct_words & user_words
    similarity = len(overlap) / len(correct_words)
    return AnswerCheckOut(correct=similarity >= 0.5, similarity=round(similarity, 2))

@api.post("/summarize-url", response_model=UrlSummaryOut)
def summarize_url_api(payload: UrlIn):
    # 1) 본문 추출
    try:
        content = get_article_text(str(payload.url))
    except HTTPException as he:
        raise HTTPException(status_code=he.status_code, detail=f"[본문 추출 실패] {he.detail}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[본문 추출 예외] {e}")

    if not content or len(content.strip()) < 50:
        raise HTTPException(status_code=404, detail="[본문 없음] 충분한 본문을 찾지 못했습니다.")

    # 2) 3줄 요약
    try:
        summary = summarize_3lines(content)
        if not summary:
            raise HTTPException(status_code=500, detail="요약 생성에 실패했습니다.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[요약 예외] {e}")

    # 3) 키워드
    keywords = extract_keywords(content)

    return UrlSummaryOut(
        summary_3lines=summary,
        keywords=keywords,
        meta={"url": str(payload.url), "content_len": str(len(content))}
    )

# 라우터 등록
app.include_router(api)

# ─────────────────────────────────────────────────────────────────────
# 로컬 실행 (직접 실행 시)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # 포트 충돌·권한 이슈 피하려고 8765로 설정 (필요 시 변경 가능)
    uvicorn.run(app, host="127.0.0.1", port=8765)
