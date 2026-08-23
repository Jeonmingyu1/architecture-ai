import streamlit as st
import pandas as pd
from openai import OpenAI
import csv
import os
import re
import random

# 1. 페이지 설정
st.set_page_config(page_title="건축기사 AI 채점 로봇", layout="wide")
st.title("🏗️ 건축기사 AI 학습 & 채점 시스템")

# 2. Gemini API 키 설정
client = OpenAI(
    api_key=st.secrets["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 3. CSV 데이터 불러오기
try:
    df = pd.read_csv('data.csv', encoding='cp949')
except FileNotFoundError:
    st.error("⚠️ 'data.csv' 파일이 없습니다. ai_project 폴더 안에 data.csv 파일을 먼저 만들어주세요!")
    st.stop()

# --- RAG 검색 함수 ---
def search_related_context(df, current_keyword, current_sub, current_question):
    related_rows = df[
        ((df['개념 키워드'].str.contains(current_keyword, na=False)) | (df['중단원'] == current_sub)) & 
        (df['문제 내용'] != current_question)
    ].head(2)
    
    context_text = ""
    if not related_rows.empty:
        context_text += "[참고할 유사 기출 및 연계 개념 (RAG Context)]\n"
        for idx, row in related_rows.iterrows():
            context_text += f"- 관련 문제: {row['문제 내용']}\n"
            context_text += f"- 모범 답안: {row['모범 답안']}\n"
            context_text += f"- 해설: {row['해설']}\n\n"
            
    return context_text

# --- 점수 추출 헬퍼 함수 ---
def extract_score(result_text):
    match = re.search(r'(?:최종\s*점수|점수)[\s:]*([0-9]{1,3})점?', result_text)
    if match:
        return int(match.group(1))
    match_any = re.search(r'\b([0-9]{1,3})\b', result_text)
    if match_any:
        val = int(match_any.group(1))
        if 0 <= val <= 100:
            return val
    return 0

# 4. 상단 탭 나누기
tab1, tab2 = st.tabs(["📝 기출문제 풀기 & AI 채점", "📊 나의 학습 분석 & 오답노트"])

# ==================== [탭 1: 문제 풀이 화면] ====================
with tab1:
    st.sidebar.header("📚 학습 챕터 선택")
    major_categories = df['대단원'].unique().tolist()
    selected_major = st.sidebar.selectbox("대단원을 선택하세요:", major_categories)

    sub_df = df[df['대단원'] == selected_major]
    sub_categories = sub_df['중단원'].unique().tolist()
    selected_sub = st.sidebar.selectbox("중단원을 선택하세요:", sub_categories)

    # 챕터가 바뀌면 이전 랜덤 문제 기록 초기화
    if 'prev_sub' not in st.session_state or st.session_state['prev_sub'] != selected_sub:
        st.session_state['prev_sub'] = selected_sub
        if 'current_random_q' in st.session_state:
            del st.session_state['current_random_q']

    filtered_df = sub_df[sub_df['중단원'] == selected_sub]
    all_questions = filtered_df['문제 내용'].tolist()

    st.sidebar.divider()
    st.sidebar.info(f"현재 챕터의 전체 문제 수: **{len(all_questions)}개**")

    if not all_questions:
        st.warning("해당 챕터에 등록된 문제가 없습니다.")
    else:
        mode = st.radio("🔍 문제 풀이 방식을 선택하세요:", ["목록에서 직접 선택하기", "🎲 랜덤으로 문제 뽑기"], horizontal=True)
        st.divider()

        if mode == "목록에서 직접 선택하기":
            if 'current_random_q' in st.session_state:
                del st.session_state['current_random_q']
            selected_q = st.selectbox("📌 문제를 선택하세요:", all_questions)
        else:
            if 'current_random_q' not in st.session_state or st.button("🎲 다른 랜덤 문제 뽑기"):
                st.session_state['current_random_q'] = random.choice(all_questions)
            
            selected_q = st.session_state['current_random_q']
            st.info(f"🎲 랜덤 출제된 문제입니다: **{selected_q}**")

        match_rows = filtered_df[filtered_df['문제 내용'] == selected_q]
        if match_rows.empty:
            selected_q = all_questions[0]
            row_data = filtered_df.iloc[0]
        else:
            row_data = match_rows.iloc[0]

        keyword = row_data['개념 키워드']
        correct_answer = row_data['모범 답안']
        explanation = row_data['해설']

        user_answer = st.text_area("✍️ 정답을 서술해 주세요:")

        if st.button("AI 채점 및 분석 요청"):
            if not user_answer:
                st.warning("답안을 입력해주세요!")
            else:
                with st.spinner("🤖 채점 중입니다..."):
                    
                    rag_context = search_related_context(df, keyword, selected_sub, selected_q)
                    
                    prompt = f"""
                    너는 건축기사 실기 국가자격증 시험의 수석 채점관이자 시공 기술사야.
                    아래의 [참고 기출 데이터(RAG)]와 [본 문제 모범 답안]을 기준으로 학생의 답안을 매우 엄격하고 객관적으로 채점해줘.

                    {rag_context}

                    [현재 채점할 문제]: {selected_q}
                    [필수 모범 답안]: {correct_answer}
                    [상세 해설]: {explanation}
                    [학생이 작성한 답안]: {user_answer}

                    채점 지침:
                    1. 모범 답안에 포함된 '핵심 키워드'가 학생 답안에 정확히 들어갔는지 엄격하게 평가하여 0~100점의 점수를 부여할 것.
                    2. 감점 요인이 있다면 어떤 용어나 개념이 누락되었는지 명시할 것.
                    3. 과거 유사 기출문제(RAG 참고 자료)의 출제 경향과 연결지어, 학생이 이 개념을 완벽히 이해했는지 입체적으로 피드백할 것.

                    반드시 아래 형식으로만 출력할 것 (특히 1번 항목은 '1. 최종 점수: XX점' 형태로 명시할 것):
                    1. 최종 점수: (예: 80점)
                    2. 키워드 포함 여부: (누락된 핵심 키워드 지적)
                    3. 채점 상세 평가: (정답 및 오답 이유 분석)
                    4. 취약점 및 연계 학습 피드백: (유사 기출 연계 조언)
                    """

                    response = client.chat.completions.create(
                        model="gemini-3.6-flash", 
                        messages=[{"role": "user", "content": prompt}]
                    )

                    result = response.choices[0].message.content
                    score = extract_score(result)

                    st.subheader("📌 AI 채점 및 피드백 결과")
                    st.write(result)

                    file_name = 'results.csv'
                    file_exists = os.path.isfile(file_name)

                    with open(file_name, mode='a', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['선택한문제', '학생답안', '점수', 'AI채점결과'])
                        
                        clean_result = result.replace('\n', ' ')
                        writer.writerow([selected_q, user_answer, score, clean_result])

                    st.success("📁 채점 결과가 학습 분석 데이터에 안전하게 저장되었습니다!")

# ==================== [탭 2: 학습 분석 및 오답노트 화면] ====================
with tab2:
    st.header("📈 나의 학습 기록 및 오답 분석 대시보드")
    
    results_file = 'results.csv'
    
    if not os.path.isfile(results_file):
        st.info("💡 아직 풀고 저장한 문제 기록이 없습니다. '기출문제 풀기' 탭에서 문제를 풀고 채점을 진행해 보세요!")
    else:
        res_df = pd.read_csv(results_file, encoding='utf-8-sig')
        total_solved = len(res_df)
        
        if '점수' in res_df.columns and total_solved > 0:
            avg_score = res_df['점수'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="총 풀이 횟수", value=f"{total_solved}회")
            with col2:
                st.metric(label="평균 취득 점수", value=f"{avg_score:.1f}점")
            with col3:
                status = "🎯 합격 안정권" if avg_score >= 60 else "⚠️ 집중 학습 필요"
                st.metric(label="현재 학습 성취도", value=status)
            
            st.divider()
            st.subheader("📊 학습 점수 추이 그래프")
            st.line_chart(res_df['점수'])
            
            st.divider()
            st.subheader("📋 오답노트 및 전체 학습 기록")
            only_weak = st.checkbox("❌ 60점 미만 오답 문제만 모아서 보기")
            
            display_df = res_df[res_df['점수'] < 60] if only_weak else res_df
            st.dataframe(display_df, use_container_width=True)
            
        else:
            st.metric(label="총 풀이 횟수", value=f"{total_solved}회")
            st.dataframe(res_df, use_container_width=True)
        
        st.divider()
        if st.button("🗑️ 학습 기록 초기화하기"):
            if os.path.isfile(results_file):
                os.remove(results_file)
                st.rerun()
