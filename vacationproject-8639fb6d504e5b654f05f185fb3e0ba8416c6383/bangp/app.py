# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import re
import json
from fastapi import HTTPException

app = FastAPI()

# ----------------------
# 데이터 모델
# ----------------------
class SummaryIn(BaseModel):
    summary: str

class QuestionOut(BaseModel):
    question: str
    answer: str  # 정답(요약에서 뽑은 문장)

class AnswerCheckIn(BaseModel):
    user_answer: str
    correct_answer: str

class AnswerCheckOut(BaseModel):
    correct: bool
    similarity: float  # 0~1 사이 (정답과 비슷한 정도)

SUMMARY_FILE = "summary.json"

def load_summary():
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

# ----------------------
# 간단한 문장 추출 함수
# ----------------------
def get_first_sentence(text: str) -> str:
    sentences = re.split(r'[.?!\n]', text)  # 마침표, 물음표, 줄바꿈으로 분리
    for s in sentences:
        if s.strip():
            return s.strip()
    return text.strip()

# ----------------------
# 질문 생성 API
# ----------------------
@app.post("/generate-question", response_model=QuestionOut)
def generate_question(data: SummaryIn):
    first_sentence = get_first_sentence(data.summary)
    question = f"{first_sentence}에 대해 설명해주세요."
    return QuestionOut(question=question, answer=first_sentence)

# ----------------------
# 답안 채점 API
# ----------------------
@app.post("/check-answer", response_model=AnswerCheckOut)
def check_answer(data: AnswerCheckIn):
    # 소문자 변환 + 특수문자 제거
    def clean_text(t):
        return re.sub(r'[^가-힣a-zA-Z0-9\s]', '', t.lower())

    correct_clean = clean_text(data.correct_answer)
    user_clean = clean_text(data.user_answer)

    # 단어 단위 비교
    correct_words = set(correct_clean.split())
    user_words = set(user_clean.split())

    if not correct_words:
        return AnswerCheckOut(correct=False, similarity=0.0)

    overlap = correct_words & user_words
    similarity = len(overlap) / len(correct_words)

    return AnswerCheckOut(
        correct=similarity >= 0.5,  # 50% 이상 같으면 정답 처리
        similarity=round(similarity, 2)
    )

# ----------------------
# 요약본 조회 API
# ----------------------
@app.get("/summary")
def get_summary():
    summary = load_summary()
    return {"summary": summary}

# ----------------------
# 요약본 수정 API
# ----------------------
@app.post("/summary")
def update_summary(data: SummaryIn):
    save_summary(data.summary)
    return {"message": "요약이 저장되었습니다.", "summary": data.summary}

# ----------------------
# 서버 상태 확인
# ----------------------
@app.get("/")
def root():
    return {"message": "요약 기반 질문 생성 & 채점 API입니다."}
