from transformers import pipeline
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# 요약 모델 로드(koBART)
summ = pipeline("summarization", model="gogamza/kobart-summarization", tokenizer="gogamza/kobart-summarization", device=-1)
print("summarizer ok")

# 키워드 모델 로드
sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
kw = KeyBERT(model=sbert)
print("keybert ok")

text = "정부는 인공지능 산업을 육성하기 위한 종합 대책을 발표했다. 스타트업 지원과 반도체 인프라 투자 확대가 주요 내용이다."
print(summ(text, max_length=60, min_length=30, do_sample=False)[0]["summary_text"])
print(kw.extract_keywords(text, keyphrase_ngram_range=(1,2), top_n=5))
