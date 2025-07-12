# 🚀 뉴스팩토리 - AI 기반 기사 분석 플랫폼 (konlpy 버전)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os
import time
import hashlib
from pathlib import Path
import requests
import pickle
import json
from collections import Counter, defaultdict
import subprocess
import sys
import calendar

# ──────────── konlpy Okt 로드 ────────────
from konlpy.tag import Okt
okt = Okt()

# 자동 패키지 설치 함수
def install_package(package):
    """패키지 자동 설치"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

# 머신러닝 라이브러리 import
def safe_import():
    """개선된 안전한 라이브러리 import"""
    global TfidfVectorizer, cosine_similarity
    # 필수 라이브러리
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        st.success("✅ 머신러닝 라이브러리 로드 성공")
    except ImportError:
        st.warning("⚠️ scikit-learn이 설치되지 않았습니다.")
        TfidfVectorizer = None
        cosine_similarity = None

# 라이브러리 초기화
safe_import()

# 구글 시트 유틸리티 모듈
try:
    from google_sheets_utils_deploy import (
        load_data_from_google_sheets, preprocess_dataframe,
        test_connection, get_google_sheets_client, get_connection_status
    )
    SHEETS_UTILS_AVAILABLE = True
except ImportError:
    SHEETS_UTILS_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="🚀 뉴스팩토리",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 전역 설정
PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
APP_PASSWORD = st.secrets["APP_PASSWORD"]

# 캐시 설정
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
DATA_CACHE_DIR = CACHE_DIR / "data"
DATA_CACHE_DIR.mkdir(exist_ok=True)
ANALYSIS_CACHE_DIR = CACHE_DIR / "analysis"
ANALYSIS_CACHE_DIR.mkdir(exist_ok=True)
API_CACHE_DIR = CACHE_DIR / "api_calls"
API_CACHE_DIR.mkdir(exist_ok=True)

# 어두운 테마 CSS 스타일
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(90deg, #0f0f0f 0%, #1a1a1a 100%);
        color: #ffffff;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .planning-item {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# 캐싱 시스템
# ================================
def get_cache_key(data_type, industry, date_str=None):
    """캐시 키 생성"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    key_string = f"{data_type}_{industry}_{date_str}"
    return hashlib.md5(key_string.encode()).hexdigest()

def save_to_cache(cache_type, key, data):
    """캐시에 데이터 저장"""
    try:
        if cache_type == "data":
            cache_file = DATA_CACHE_DIR / f"{key}.pkl"
        elif cache_type == "analysis":
            cache_file = ANALYSIS_CACHE_DIR / f"{key}.pkl"
        else:
            cache_file = API_CACHE_DIR / f"{key}.pkl"
        
        cache_data = {
            'data': data,
            'timestamp': time.time(),
            'date': datetime.now().strftime("%Y-%m-%d")
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        return True
    except Exception as e:
        st.error(f"캐시 저장 실패: {str(e)}")
        return False

def load_from_cache(cache_type, key, max_age_hours=24):
    """캐시에서 데이터 로드"""
    try:
        if cache_type == "data":
            cache_file = DATA_CACHE_DIR / f"{key}.pkl"
        elif cache_type == "analysis":
            cache_file = ANALYSIS_CACHE_DIR / f"{key}.pkl"
        else:
            cache_file = API_CACHE_DIR / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
        
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # 캐시 유효성 검사
        cache_age = time.time() - cache_data['timestamp']
        if cache_age > max_age_hours * 3600:
            return None
        
        return cache_data['data']
    except:
        return None

# ================================
# API 호출
# ================================
def call_perplexity_api_cached(prompt, model="sonar-pro", max_age_hours=24):
    """캐싱된 Perplexity API 호출"""
    try:
        # 캐시 키 생성
        cache_key = hashlib.md5(f"{prompt}_{model}".encode()).hexdigest()
        
        # 캐시에서 먼저 확인
        cached_result = load_from_cache("api", cache_key, max_age_hours)
        if cached_result:
            st.info("🔄 캐시에서 AI 분석 결과를 불러왔습니다.")
            return cached_result
        
        # API 호출
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 뉴스 전문 분석가입니다. 주어진 기사 데이터를 분석하여 핵심 트렌드와 중요한 키워드를 한국어로 간결하게 요약해주세요."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            api_result = result["choices"][0]["message"]["content"]
            
            # 결과를 캐시에 저장
            save_to_cache("api", cache_key, api_result)
            st.success("🆕 새로운 AI 분석을 완료했습니다.")
            return api_result
        else:
            st.error(f"API 오류: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"API 호출 실패: {str(e)}")
        return None

# ================================
# 🔧 NEW UTILS (새로운 기능 함수들)
# ================================
def compile_articles_text(articles, max_each=3):
    """유사 기사 리스트 → Prompt용 텍스트 문자열"""
    texts = []
    for art in articles[:max_each]:
        title = art.get("제목", "")
        body = art.get("주요내용", "")
        if pd.notna(title):
            texts.append(f"제목: {title}\n내용: {str(body)[:300]}")
    return "\n".join(texts)

def generate_monthly_ai_plans(insight_dict, df, industry="전체"):
    """
    주차별 핵심문장 + 유사 기사들을 묶어 Perplexity API로 'AI 추천 기획 아이템' 생성
    """
    if not insight_dict or df.empty:
        return "데이터가 부족해 기획 아이템을 생성할 수 없습니다."
    
    plan_source_blocks = []
    for week, sentence in insight_dict.get("weekly_insights", {}).items():
        if not sentence:
            continue
        
        sims = find_similar_articles_to_insight(df, sentence, 7)
        sim_text = compile_articles_text([s["article"] for s in sims])
        
        plan_source_blocks.append(
            f"[{week}] 핵심문장: {sentence}\n관련 기사:\n{sim_text}"
        )
    
    prompt = f"""
    당신은 한국 주요 일간지의 {industry if industry!='전체' else ''} 담당 부장입니다.
    아래 주차별 핵심 트렌드와 관련 기사들을 분석해 다음 달 취재 방향을 제시할 'AI 추천 기획 아이템' 5건을 작성하세요.
    
    [요청 형식]
    제시 문장(기사 제목 후보) - 한 줄 이유
    
    [분석 대상 데이터]
    {os.linesep.join(plan_source_blocks[:3])}
    """
    
    return call_perplexity_api_cached(prompt, max_age_hours=6)

def generate_custom_topic_brief(df, industry, topic, weight_threshold=0.45):
    """
    업계+주제 관련 기사 추출 후 Perplexity API로 고급 브리핑 생성
    """
    if df.empty or not topic:
        return None
    
    # 업계 필터
    pool = df if industry == "전체" else df[df["업계"] == industry]
    if pool.empty:
        return None
    
    # 유사도 계산
    pool = pool.copy()
    pool["combined_text"] = pool["제목"].fillna("") + " " + pool["주요내용"].fillna("")
    
    if not TfidfVectorizer or not cosine_similarity:
        return "TF-IDF 분석 라이브러리가 필요합니다."
    
    vec = TfidfVectorizer(max_features=1500)
    try:
        mat = vec.fit_transform(pool["combined_text"].tolist() + [topic])
        sims = cosine_similarity(mat[-1:], mat[:-1]).flatten()
        
        pool["sim"] = sims
        cand = pool[(pool["sim"] > 0.1) & (pool["전체가중치"] >= weight_threshold)]
        
        if cand.empty:
            return None
        
        cand = cand.sort_values(["sim", "전체가중치"], ascending=False).head(20)
        
        joined = [
            f"제목: {r['제목']}\n내용: {str(r['주요내용'])[:300]}"
            for _, r in cand.iterrows()
        ]
        
        prompt = f"""
        당신은 한국 언론사의 {industry if industry!='전체' else ''} 담당 편집국장입니다.
        주제: {topic}
        
        아래 기사들을 바탕으로
        1) 현재 핵심 트렌드 3가지
        2) 각 트렌드별 매체 보도 특징
        3) 앞으로 발굴해야 할 심층 기획 기사 3건(제목+간단 이유)
        을 기자들에게 지시하는 구어체 메모 형식으로 작성하세요.
        
        기사 목록:
        {os.linesep.join(joined[:10])}
        """
        
        return call_perplexity_api_cached(prompt, max_age_hours=6)
        
    except:
        return "분석 중 오류가 발생했습니다."

# ================================
# 새로운 AI 기능 함수들
# ================================
def generate_today_planning_items(df, industry="전체"):
    """
    상위 가중치 기사들을 바탕으로 Perplexity API를 호출, '오늘의 기획 아이템' 3개를 제안한다.
    """
    if df.empty:
        return "분석할 데이터가 없습니다."
    
    if industry != "전체":
        df = df[df["업계"] == industry]
    
    # 가중치 상위 기사 20개까지 대상
    top_articles = (
        df.nlargest(20, "전체가중치") if "전체가중치" in df.columns else df.head(20)
    )
    
    texts = []
    for _, art in top_articles.iterrows():
        title = art.get("제목", "")
        body = art.get("주요내용", "")
        if pd.notna(title) and pd.notna(body):
            texts.append(f"제목: {title}\n내용: {body}")
        elif pd.notna(title):
            texts.append(f"제목: {title}")
    
    if not texts:
        return "분석할 기사 내용이 없습니다."
    
    prompt = f"""
    다음은 오늘의 주요 뉴스 기사들입니다. 기자들이 추가 취재하기 좋은 '기획 아이템' 3건을 제안해주세요.
    
    기사 목록:
    {os.linesep.join(texts[:15])}
    
    [답변 형식]
    1. 기획 아이템 제목:
    - 취재 포인트:
    - 예상 소스:
    
    2. 기획 아이템 제목:
    - 취재 포인트:
    - 예상 소스:
    
    3. 기획 아이템 제목:
    - 취재 포인트:
    - 예상 소스:
    """
    
    return (
        call_perplexity_api_cached(prompt, max_age_hours=6)
        or "기획 아이템 생성에 실패했습니다."
    )

def generate_monthly_weekly_insights(df, target_month=None):
    """
    월별·주차별 핵심 문장 생성(PPLX API) → dict 반환
    {
        'monthly_insight': str,
        'weekly_insights': {주차: str, …}
    }
    """
    if df.empty:
        return {}
    
    if target_month and "월" in df.columns:
        df = df[df["월"] == target_month]
    
    if df.empty:
        return {}
    
    # 월 전체 핵심 문장
    month_txt = " ".join(
        (
            f"{r.get('제목','')} {r.get('주요내용','')}"
            for _, r in df.head(50).iterrows()
        )
    )
    
    mon_prompt = f"""
    다음은 {target_month or '이번 달'} 주요 뉴스입니다.
    핵심 이슈를 한 문장으로 요약해 주세요.
    
    뉴스 내용:
    {month_txt}
    """
    
    monthly_insight = call_perplexity_api_cached(mon_prompt, max_age_hours=12)
    
    # 주차별
    weekly_insights = {}
    if "주차" in df.columns:
        for week in sorted(df["주차"].dropna().unique()):
            w_df = df[df["주차"] == week]
            week_txt = " ".join(
                (
                    f"{r.get('제목','')} {r.get('주요내용','')}"
                    for _, r in w_df.head(20).iterrows()
                )
            )
            
            w_prompt = f"""
            다음은 {week} 주요 뉴스입니다.
            핵심 이슈를 한 문장으로 요약해 주세요.
            
            뉴스 내용:
            {week_txt}
            """
            
            weekly_insights[week] = call_perplexity_api_cached(
                w_prompt, max_age_hours=24
            )
    
    return {"monthly_insight": monthly_insight, "weekly_insights": weekly_insights}

def find_similar_articles_to_insight(df, insight, top_n=5):
    """
    핵심 문장(insight) 과 유사도가 높은 기사 n개 반환
    """
    if df.empty or not insight:
        return []
    
    df = df.copy()
    df["combined_text"] = df.apply(
        lambda r: f"{r.get('제목','')} {r.get('주요내용','')}", axis=1
    )
    df = df[df["combined_text"].str.strip() != ""]
    
    if df.empty or not (TfidfVectorizer and cosine_similarity):
        return []
    
    try:
        vec = TfidfVectorizer(max_features=1000)
        mat = vec.fit_transform(df["combined_text"].tolist() + [insight])
        sims = cosine_similarity(mat[-1:], mat[:-1]).flatten()
        
        idxs = sims.argsort()[::-1][:top_n]
        return [
            {"article": df.iloc[i], "similarity": sims[i]}
            for i in idxs if sims[i] > 0.1
        ]
    except:
        return []

# ================================
# konlpy 기반 키워드 추출 (soynlp 대체)
# ================================
def extract_keywords_okt(text_list, top_n=15):
    """konlpy Okt를 사용한 키워드 추출"""
    try:
        if not text_list:
            return []
        
        # 텍스트 전처리
        if isinstance(text_list, str):
            text_list = [text_list]
        
        # 텍스트 정제
        cleaned_texts = []
        for text in text_list:
            if pd.notna(text):
                # 기본 정제
                text = str(text).strip()
                text = re.sub(r'[^\w\s가-힣]', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                if len(text) > 10:  # 너무 짧은 텍스트 제외
                    cleaned_texts.append(text)
        
        if not cleaned_texts:
            return []
        
        # 전체 텍스트 결합
        corpus = ' '.join(cleaned_texts)
        
        # 명사 추출
        nouns = okt.nouns(corpus)
        
        # 명사 빈도 계산
        noun_freq = Counter(nouns)
        
        # 불용어 제거
        stopwords = {
            '기자', '사진', '제공', '관련', '업계', '시장', '분야', '회사', '기업',
            '발표', '공개', '설명', '보도', '뉴스', '기사', '오늘', '어제', '내일',
            '최근', '현재', '당시', '이번', '다음', '지난', '국내', '해외', '전국',
            '것', '수', '등', '및', '또', '더', '이', '그', '을', '를', '의', '가',
            '말했다', '밝혔다', '전했다', '했다', '있다', '없다', '한다', '된다'
        }
        
        # 최종 키워드 선별
        final_keywords = []
        for word, freq in noun_freq.most_common():
            if (word not in stopwords and 
                not word.isdigit() and 
                len(word) >= 2 and len(word) <= 10 and
                re.search(r'[가-힣]', word)):
                final_keywords.append((word, freq))
        
        return final_keywords[:top_n]
        
    except Exception as e:
        st.error(f"키워드 추출 실패: {str(e)}")
        return extract_keywords_simple(text_list, top_n)

def extract_keywords_simple(text_list, top_n=10):
    """백업용 간단한 키워드 추출"""
    try:
        if isinstance(text_list, str):
            text_list = [text_list]
        
        all_text = ' '.join([str(text) for text in text_list if pd.notna(text)])
        
        # 간단한 토큰화
        words = re.findall(r'[가-힣]{2,6}', all_text)
        
        # 불용어 제거
        stopwords = {
            '기자', '사진', '제공', '관련', '업계', '시장', '분야', '회사', '기업',
            '발표', '공개', '설명', '보도', '뉴스', '기사', '오늘', '어제', '내일'
        }
        
        keywords = [word for word in words if word not in stopwords]
        word_freq = Counter(keywords)
        
        return word_freq.most_common(top_n)
    except:
        return []

# 기존 extract_keywords_mecab 함수 대체
def extract_keywords_mecab(text, top_n=10):
    """키워드 추출 - konlpy Okt 사용"""
    if isinstance(text, str):
        return extract_keywords_okt([text], top_n)
    elif isinstance(text, list):
        return extract_keywords_okt(text, top_n)
    else:
        return extract_keywords_okt([str(text)], top_n)

# ================================
# 업계별 키워드 추출 기능
# ================================
def extract_industry_keywords(df, industry):
    """업계별 특화 키워드 추출"""
    if industry != '전체':
        industry_articles = df[df['업계'] == industry]
    else:
        industry_articles = df
    
    titles = industry_articles['제목'].dropna().tolist()
    
    # konlpy Okt로 업계 특화 키워드 추출
    keywords = extract_keywords_okt(titles, 20)
    
    # 업계별 가중치 조정
    industry_weights = {
        '자동차': ['전기차', '자율주행', '배터리', '모빌리티', '전동화'],
        'IT/전자': ['AI', '반도체', '클라우드', '디지털', '인공지능'],
        '금융': ['핀테크', '가상화폐', '블록체인', '투자', '디지털뱅킹'],
        '바이오': ['백신', '치료제', '임상', '신약', '의료'],
        '화학/에너지': ['친환경', '탄소중립', '재생에너지', '수소'],
        '항공/운송': ['물류', '배송', '모빌리티', '운송'],
        '건설/부동산': ['개발', '분양', '재건축', '리모델링'],
        '유통/소비재': ['이커머스', '온라인', '배송', '소비'],
        '엔터테인먼트/미디어': ['콘텐츠', '플랫폼', '스트리밍', 'OTT'],
        '게임': ['메타버스', '게임', '플랫폼', '모바일'],
        '식품/음료': ['건강', '프리미엄', '친환경', '유기농']
    }
    
    if industry in industry_weights:
        boost_keywords = industry_weights[industry]
        enhanced_keywords = []
        for word, score in keywords:
            if any(boost in word for boost in boost_keywords):
                enhanced_keywords.append((word, score * 1.5))
            else:
                enhanced_keywords.append((word, score))
        return sorted(enhanced_keywords, key=lambda x: x[1], reverse=True)
    
    return keywords

# ================================
# 네이버 링크 하이퍼링크 처리 함수 (핵심 수정 부분)
# ================================
def get_naver_link_column(df):
    """구글 시트에서 네이버 링크 컬럼명을 찾는 함수"""
    possible_columns = ['네이버링크', '네이버 링크', '네이버URL', '네이버 URL', 'naver_link', 'naver_url']
    for col in possible_columns:
        if col in df.columns:
            return col
    
    # 유사한 컬럼명 찾기
    for col in df.columns:
        if '네이버' in str(col) and ('링크' in str(col) or 'URL' in str(col)):
            return col
    
    return None

def get_media_link_column(df):
    """구글 시트에서 매체링크 컬럼명을 찾는 함수"""
    possible_columns = ['매체링크', '매체 링크', '매체URL', '매체 URL', 'media_link', 'media_url']
    for col in possible_columns:
        if col in df.columns:
            return col
    
    # 유사한 컬럼명 찾기
    for col in df.columns:
        if '매체' in str(col) and ('링크' in str(col) or 'URL' in str(col)):
            return col
    
    return None

def create_article_hyperlink(title, naver_url=None, media_url=None):
    """기사 제목에 하이퍼링크를 생성하는 함수"""
    if pd.isna(title):
        title = "제목 없음"
    else:
        title = str(title).strip()
    
    # 1순위: 네이버링크 사용
    if pd.notna(naver_url) and str(naver_url).strip() and str(naver_url).strip().lower() not in ['nan', '', 'none']:
        naver_url = str(naver_url).strip()
        if not naver_url.startswith(('http://', 'https://')):
            naver_url = 'https://' + naver_url
        return f'📰 [{title}]({naver_url})'
    
    # 2순위: 매체링크 사용
    if pd.notna(media_url) and str(media_url).strip() and str(media_url).strip().lower() not in ['nan', '', 'none']:
        media_url = str(media_url).strip()
        if not media_url.startswith(('http://', 'https://')):
            media_url = 'https://' + media_url
        return f'📰 [{title}]({media_url})'
    
    return f"📄 {title}"

def display_article_with_link(article, df=None):
    """기사를 하이퍼링크와 함께 표시하는 함수"""
    title = article.get('제목', '제목 없음')
    
    # 네이버 링크 컬럼명 자동 감지
    if df is not None:
        naver_col = get_naver_link_column(df)
        naver_url = article.get(naver_col) if naver_col else None
        
        # 매체 링크 컬럼명 자동 감지
        media_col = get_media_link_column(df)
        media_url = article.get(media_col) if media_col else None
    else:
        naver_url = article.get('네이버링크') or article.get('네이버 URL')
        media_url = article.get('매체링크') or article.get('매체 URL')
    
    # 하이퍼링크가 적용된 제목 생성
    linked_title = create_article_hyperlink(title, naver_url, media_url)
    return linked_title

# ================================
# PyArrow 오류 해결용 안전한 테이블 표시 함수
# ================================
def safe_display_dataframe(df, title="데이터"):
    """PyArrow 오류를 피한 안전한 데이터프레임 표시"""
    try:
        # 먼저 st.dataframe 시도
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        # PyArrow 오류 시 대안 표시
        st.warning(f"⚠️ 표 표시 오류 (PyArrow): {str(e)}")
        st.markdown(f"### 📊 {title}")
        
        # HTML 테이블로 대체
        html_table = "<table style='width:100%; border-collapse: collapse;'>"
        html_table += "<tr style='background-color: #f0f0f0;'>"
        for col in df.columns:
            html_table += f"<th style='border: 1px solid #ddd; padding: 8px;'>{col}</th>"
        html_table += "</tr>"
        
        for _, row in df.head(10).iterrows():
            html_table += "<tr>"
            for col in df.columns:
                html_table += f"<td style='border: 1px solid #ddd; padding: 8px;'>{row[col]}</td>"
            html_table += "</tr>"
        
        html_table += "</table>"
        st.markdown(html_table, unsafe_allow_html=True)
        
        st.markdown(f"*... 총 {len(df)}개 행 중 10개만 표시*")

# ================================
# 로그인 함수
# ================================
def check_password():
    """기자용 로그인 화면"""
    if st.session_state.get('authenticated', False):
        return True
    
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 20px; margin-bottom: 2rem; color: white;'>
        <h2>🔐 기자님들을 위한 전용 서비스입니다</h2>
        <p>AI 기반 기사 분석 플랫폼</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 로그인 폼
    with st.form("login_form"):
        password = st.text_input("🔑 접속 비밀번호", type="password")
        submitted = st.form_submit_button("로그인", use_container_width=True)
        
        if submitted:
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.success("✅ 로그인 성공! 잠시만 기다려주세요...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ 비밀번호가 올바르지 않습니다.")
    
    return False

# ================================
# 메인 애플리케이션 로직
# ================================
def main():
    """메인 애플리케이션 함수"""
    
    # 로그인 확인
    if not check_password():
        st.stop()
    
    # 메인 헤더
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 20px; margin-bottom: 2rem; color: white;'>
        <h1>🚀 뉴스팩토리 AI 기반 기사 분석 플랫폼</h1>
        <p>konlpy 기반 한국어 처리 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 연결 상태 확인
    if SHEETS_UTILS_AVAILABLE:
        status = get_connection_status()
        if status['connected']:
            st.success("✅ 구글 시트 연결 성공")
        else:
            st.error(f"❌ 구글 시트 연결 실패: {status['message']}")
            st.stop()
    else:
        st.error("❌ 구글 시트 유틸리티 모듈을 불러올 수 없습니다.")
        st.stop()
    
    # 데이터 로드
    with st.spinner("🔄 데이터를 불러오는 중..."):
        df = load_data_from_google_sheets()
        
        if df.empty:
            st.error("❌ 데이터를 불러올 수 없습니다.")
            st.stop()
        
        # 전처리
        df = preprocess_dataframe(df)
        
        if df.empty:
            st.error("❌ 데이터 전처리에 실패했습니다.")
            st.stop()
    
    st.success(f"✅ 총 {len(df)}개 기사 데이터 로드 완료")
    
    # 사이드바 - 업계 선택
    st.sidebar.header("🔍 필터 설정")
    
    industries = ["전체"] + sorted(df["업계"].dropna().unique().tolist())
    selected_industry = st.sidebar.selectbox("업계 선택", industries)
    
    # 필터된 데이터
    filtered_df = df if selected_industry == "전체" else df[df["업계"] == selected_industry]
    
    # 메인 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📊 데이터 분석", "🔍 키워드 분석", "🤖 AI 인사이트", "📈 트렌드 분석"])
    
    with tab1:
        st.header("📊 기사 데이터 분석")
        
        # 기본 통계
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 기사 수", len(filtered_df))
        
        with col2:
            if "전체가중치" in filtered_df.columns:
                avg_weight = filtered_df["전체가중치"].mean()
                st.metric("평균 가중치", f"{avg_weight:.3f}")
            else:
                st.metric("평균 가중치", "미계산")
        
        with col3:
            if "월" in filtered_df.columns:
                month_count = filtered_df["월"].nunique()
                st.metric("대상 월", f"{month_count}개월")
            else:
                st.metric("대상 월", "미분류")
        
        with col4:
            if "업계" in filtered_df.columns:
                industry_count = filtered_df["업계"].nunique()
                st.metric("업계 수", f"{industry_count}개")
            else:
                st.metric("업계 수", "미분류")
        
        # 상위 기사 표시
        st.subheader("🏆 상위 가중치 기사")
        
        if "전체가중치" in filtered_df.columns:
            top_articles = filtered_df.nlargest(10, "전체가중치")
        else:
            top_articles = filtered_df.head(10)
        
        for idx, (_, article) in enumerate(top_articles.iterrows(), 1):
            with st.expander(f"{idx}. {article.get('제목', '제목 없음')}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**매체**: {article.get('매체', '정보 없음')}")
                    st.write(f"**날짜**: {article.get('보도날짜', '정보 없음')}")
                    st.write(f"**업계**: {article.get('업계', '정보 없음')}")
                    if pd.notna(article.get('주요내용')):
                        st.write(f"**내용**: {str(article.get('주요내용'))[:200]}...")
                
                with col2:
                    if "전체가중치" in article:
                        st.metric("가중치", f"{article['전체가중치']:.3f}")
                    
                    # 링크 표시
                    linked_title = display_article_with_link(article, df)
                    st.markdown(linked_title)
    
    with tab2:
        st.header("🔍 키워드 분석")
        
        # 키워드 추출
        st.subheader("📝 주요 키워드")
        
        with st.spinner("🔄 키워드 분석 중..."):
            # 제목 기반 키워드 추출
            titles = filtered_df["제목"].dropna().tolist()
            keywords = extract_keywords_okt(titles, 20)
            
            if keywords:
                # 키워드 테이블
                keyword_df = pd.DataFrame(keywords, columns=["키워드", "빈도"])
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(keyword_df, use_container_width=True)
                
                with col2:
                    # 상위 10개 키워드 차트
                    if len(keywords) >= 10:
                        chart_data = pd.DataFrame(keywords[:10], columns=["키워드", "빈도"])
                        st.bar_chart(chart_data.set_index("키워드"))
            else:
                st.warning("⚠️ 키워드를 추출할 수 없습니다.")
        
        # 업계별 키워드
        if selected_industry != "전체":
            st.subheader(f"🏢 {selected_industry} 업계 특화 키워드")
            
            industry_keywords = extract_industry_keywords(df, selected_industry)
            
            if industry_keywords:
                industry_keyword_df = pd.DataFrame(industry_keywords[:15], columns=["키워드", "가중점수"])
                st.dataframe(industry_keyword_df, use_container_width=True)
    
    with tab3:
        st.header("🤖 AI 인사이트")
        
        # 오늘의 기획 아이템
        st.subheader("💡 오늘의 기획 아이템")
        
        if st.button("🚀 AI 기획 아이템 생성", use_container_width=True):
            with st.spinner("🤖 AI가 기획 아이템을 생성하는 중..."):
                planning_items = generate_today_planning_items(filtered_df, selected_industry)
                
                st.markdown(f"""
                <div class="insight-box">
                    <h3>💡 AI 추천 기획 아이템</h3>
                    <p>{planning_items}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 월별/주차별 인사이트
        st.subheader("📅 월별/주차별 인사이트")
        
        if "월" in filtered_df.columns:
            available_months = sorted(filtered_df["월"].dropna().unique())
            if available_months:
                selected_month = st.selectbox("분석할 월 선택", available_months)
                
                if st.button("🔍 월별 인사이트 생성", use_container_width=True):
                    with st.spinner("🤖 AI가 월별 인사이트를 생성하는 중..."):
                        insights = generate_monthly_weekly_insights(filtered_df, selected_month)
                        
                        if insights.get("monthly_insight"):
                            st.markdown(f"""
                            <div class="insight-box">
                                <h3>📊 {selected_month} 월별 핵심 인사이트</h3>
                                <p>{insights['monthly_insight']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if insights.get("weekly_insights"):
                            st.subheader("📅 주차별 인사이트")
                            for week, insight in insights["weekly_insights"].items():
                                if insight:
                                    st.markdown(f"""
                                    <div class="planning-item">
                                        <strong>{week}:</strong> {insight}
                                    </div>
                                    """, unsafe_allow_html=True)
        
        # 커스텀 주제 브리핑
        st.subheader("🎯 커스텀 주제 브리핑")
        
        custom_topic = st.text_input("분석하고 싶은 주제를 입력하세요", placeholder="예: 전기차 시장 동향")
        
        if st.button("🔍 주제 브리핑 생성", use_container_width=True) and custom_topic:
            with st.spinner("🤖 AI가 주제 브리핑을 생성하는 중..."):
                topic_brief = generate_custom_topic_brief(filtered_df, selected_industry, custom_topic)
                
                if topic_brief:
                    st.markdown(f"""
                    <div class="insight-box">
                        <h3>🎯 {custom_topic} 브리핑</h3>
                        <p>{topic_brief}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ 해당 주제에 대한 관련 기사가 충분하지 않습니다.")
    
    with tab4:
        st.header("📈 트렌드 분석")
        
        # 월별 기사 수 추이
        if "월" in filtered_df.columns:
            st.subheader("📊 월별 기사 수 추이")
            
            monthly_counts = filtered_df["월"].value_counts().sort_index()
            st.bar_chart(monthly_counts)
        
        # 업계별 분포
        if "업계" in filtered_df.columns:
            st.subheader("🏢 업계별 기사 분포")
            
            industry_counts = filtered_df["업계"].value_counts().head(10)
            st.bar_chart(industry_counts)
        
        # 매체별 분포
        if "매체" in filtered_df.columns:
            st.subheader("📰 매체별 기사 분포")
            
            media_counts = filtered_df["매체"].value_counts().head(10)
            st.bar_chart(media_counts)
        
        # 가중치 분포
        if "전체가중치" in filtered_df.columns:
            st.subheader("⚖️ 가중치 분포")
            
            st.histogram(filtered_df["전체가중치"], bins=20)
    
    # 푸터
    st.markdown("""
    <div style='text-align: center; padding: 1rem; margin-top: 2rem; 
                background: rgba(255,255,255,0.1); border-radius: 10px;'>
        <p>🚀 뉴스팩토리 AI 기반 기사 분석 플랫폼 (konlpy 버전)</p>
        <p>✨ 전문 기자를 위한 AI 뉴스 분석 도구</p>
    </div>
    """, unsafe_allow_html=True)

# ================================
# 애플리케이션 실행
# ================================
if __name__ == "__main__":
    main()
