import streamlit as st
import pandas as pd
from openai import OpenAI
import csv
import os
import re

# 1. 페이지 설정
st.set_page_config(page_title="건축기사 AI 채점 로봇", layout="wide")
st.title("🏗️ 건축기사 AI 학습 & 채점 시스템")

# 2. Gemini API 키 설정 (스트림릿 비밀 금고에서 안전하게 불러오기)
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

# --- [개선된 RAG 핵심: 키워드 부분 일치 및 중단원 연계 검색 함수] ---
def search_related_context(df, current_keyword, current_sub, current_question):
    # 키워드가 포함되거나 같은 중단원인 데이터를 유연하게 탐색
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

# --- AI 결과에서 점수(숫자)를 추출하는 헬퍼 함수 ---
def extract_score(result_text):
    # '최종 점수: 80점' 또는 '80점' 등의 패턴을 찾아 숫자만 추출
    match = re.search(r'(?:최종\s*점수|점수)[\s:]*([0-9]{1,3})점?', result_text)
    if match:
        return int(match.group(1))
    # 만약 명시적 패턴이 없으면 텍스트 안의 첫 번째 숫자(0~100) 탐색 시도
    match_any = re.search(r'\b([0-9]{1,3})\b', result_text)
    if match_any:
        val = int(match_any.group(1))
        if 0 <= val <= 100:
            return val
    return 0 # 못 찾을 경우 기본값

# 4. 상단 탭 나누기 (탭 1: 문제 풀이 / 탭 2: 학습 분석 및 오답노트)
tab1, tab2 = st.tabs(["📝 기출문제 풀기 & AI 채점", "📊 나의 학습 분석 & 오답노트"])

# ==================== [탭 1: 문제 풀이 화면] ====================
with tab1:
    st.sidebar.header("📚 학습 챕터 선택")
    major_categories = df['대단원'].unique().tolist()
    selected_major = st.sidebar.selectbox("대단원을 선택하세요:", major_categories)

    sub_df = df[df['대단원'] == selected_major]
    sub_categories = sub_df['중단원'].unique().tolist()
    selected_sub = st.sidebar.selectbox("중단원을 선택하세요:", sub_categories)

    filtered_df = sub_df[sub_df['중단원'] == selected_sub]
    question_list = filtered_df['문제 내용'].tolist()

    st.sidebar.divider()
    st.sidebar.info(f"현재 챕터의 문제 수: **{len(question_list)}개**")

    if not question_list:
        st.warning("해당 챕터에 등록된 문제가 없습니다.")
    else:
        selected_q = st.selectbox("📌 문제를 선택하세요:", question_list)

        row_data = filtered_df[filtered_df['문제 내용'] == selected_q].iloc[0]
        keyword = row_data['개념 키워드']
        correct_answer = row_data['모범 답안']
        explanation = row_data['해설']

        user_answer = st.text_area("✍️ 정답을 서술해 주세요:")

        if st.button("AI 채점 및 분석 요청"):
            if not user_answer:
                st.warning("답안을 입력해주세요!")
            else:
                with st.spinner("🤖 채점 중입니다..."):
                    
                    # 업그레이드된 RAG 함수 호출
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
                    
                    # 점수 정량적 추출
                    score = extract_score(result)

                    st.subheader("📌 AI 채점 및 피드백 결과")
                    st.write(result)

                    # --- results.csv에 점수 컬럼 포함하여 누적 저장 ---
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
        
        # 점수 시각화 및 지표 표시
        if '점수' in res_df.columns and total_solved > 0:
            avg_score = res_df['점수'].mean()
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="총 풀이 및 채점 횟수", value=f"{total_solved}회")
            with col2:
                st.metric(label="평균 취득 점수", value=f"{avg_score:.1f}점")
            
            st.divider()
            st.subheader("📊 학습 점수 추이 그래프")
            st.line_chart(res_df['점수'])
        else:
            st.metric(label="총 풀이 및 채점 횟수", value=f"{total_solved}회")
        
        st.divider()
        st.subheader("📋 전체 채점 및 오답노트 기록")
        st.dataframe(res_df, use_container_width=True)
        
        if st.button("🗑️ 학습 기록 초기화하기"):
            if os.path.isfile(results_file):
                os.remove(results_file)
                st.rerun()
