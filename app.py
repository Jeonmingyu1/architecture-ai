import streamlit as st
import pandas as pd
from openai import OpenAI
import csv
import os
import re
import random

# 1. 페이지 설정
st.set_page_config(page_title="건축기사 AI 학습 시스템", layout="wide")
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

# 4. 상단 대메뉴 탭 (딱 2개로 분리)
tab1, tab2 = st.tabs(["📝 문제 풀이 및 AI 채점", "📊 나의 학습 기록 & 오답노트"])

# ==================== [탭 1: 문제 풀이] ====================
with tab1:
    # 상단에 깔끔하게 모드 선택 배치 (사이드바 복잡함 제거)
    col_mode1, col_mode2 = st.columns(2)
    with col_mode1:
        solve_type = st.radio("📌 학습 방식을 선택하세요:", ["📑 시험지 모드 (여러 문제 한번에 풀기)", "🔍 한 문제씩 집중 풀기"], horizontal=True)
    with col_mode2:
        num_q = st.slider("🎲 무작위 출제 문항 수", min_value=5, max_value=20, value=5, step=5)

    # 매번 새로 고칠 수 있는 랜덤 문제 세션 관리
    if 'exam_df' not in st.session_state or st.button("🔄 새로운 문제 세트 불러오기"):
        st.session_state['exam_df'] = df.sample(n=min(num_q, len(df))).reset_index(drop=True)
    
    current_df = st.session_state['exam_df']
    st.divider()

    # --- 모드 1: 시험지 모드 (여러 문제 한번에) ---
    if solve_type == "📑 시험지 모드 (여러 문제 한번에 풀기)":
        st.markdown("👇 아래 문제들을 읽고 각각 답안을 작성한 뒤, 맨 아래 **[일괄 채점하기]** 버튼을 누르세요.")
        
        user_answers = {}
        for idx, row in current_df.iterrows():
            st.markdown(f"**Q{idx+1}. [{row['대단원']}] {row['문제 내용']}**")
            ans = st.text_area(f"답안 입력 (Q{idx+1})", key=f"ans_{idx}", height=100)
            user_answers[idx] = {
                "question": row['문제 내용'],
                "correct": row['모범 답안'],
                "explanation": row['해설'],
                "user_ans": ans
            }
            st.markdown("---")

        if st.button("📝 전체 답안 일괄 채점 및 저장", type="primary"):
            with st.spinner("🤖 AI가 답안을 엄격하게 채점 중입니다..."):
                file_name = 'results.csv'
                file_exists = os.path.isfile(file_name)
                batch_results = []

                for idx, data in user_answers.items():
                    if not data["user_ans"]:
                        continue
                    
                    prompt = f"""
                    너는 건축기사 실기 수석 채점관이야.
                    [문제]: {data['question']}
                    [모범 답안]: {data['correct']}
                    [학생 답안]: {data['user_ans']}
                    
                    핵심 키워드가 포함되었는지 엄격하게 평가하여 0~100점의 점수를 부여하고 피드백을 줘.
                    반드시 아래 형식으로 출력할 것:
                    1. 최종 점수: XX점
                    2. 피드백: (간단한 평가 및 누락된 키워드)
                    """
                    
                    try:
                        response = client.chat.completions.create(
                            model="gemini-3.6-flash",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        res_text = response.choices[0].message.content
                        score = extract_score(res_text)
                    except Exception as e:
                        res_text = f"채점 오류: {str(e)}"
                        score = 0
                        
                    batch_results.append({"question": data['question'], "user_ans": data['user_ans'], "score": score, "result": res_text})
                    
                    # CSV 저장
                    with open(file_name, mode='a', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['선택한문제', '학생답안', '점수', 'AI채점결과'])
                            file_exists = True
                        writer.writerow([data['question'], data['user_ans'], score, res_text.replace('\n', ' ')])

                st.success("🎉 채점이 완료되었습니다! 아래 결과를 확인하세요.")
                for res in batch_results:
                    with st.expander(f"📌 [점수: {res['score']}점] {res['question'][:35]}..."):
                        st.markdown(f"**내 답안:** {res['user_ans']}")
                        st.markdown(f"**AI 채점 결과:**\n{res['result']}")

    # --- 모드 2: 한 문제씩 집중 풀기 ---
    else:
        q_list = current_df['문제 내용'].tolist()
        selected_q = st.selectbox("📌 풀고 싶은 문제를 선택하세요:", q_list)
        
        row_data = current_df[current_df['문제 내용'] == selected_q].iloc[0]
        
        st.info(f"**선택된 문제:** {selected_q}")
        user_ans = st.text_area("✍️ 정답을 입력하세요:", height=120)

        if st.button("🤖 AI 채점 요청", type="primary"):
            if not user_ans:
                st.warning("답안을 입력해주세요!")
            else:
                with st.spinner("채점 중..."):
                    prompt = f"""
                    너는 건축기사 실기 수석 채점관이야.
                    [문제]: {selected_q}
                    [모범 답안]: {row_data['모범 답안']}
                    [상세 해설]: {row_data['해설']}
                    [학생 답안]: {user_ans}
                    
                    핵심 키워드 포함 여부를 엄격히 평가해 0~100점의 점수를 매기고 피드백해줘.
                    반드시 아래 형식으로 출력할 것:
                    1. 최종 점수: XX점
                    2. 키워드 포함 여부: (...)
                    3. 채점 상세 평가: (...)
                    """
                    response = client.chat.completions.create(
                        model="gemini-3.6-flash",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    result_text = response.choices[0].message.content
                    score = extract_score(result_text)

                    st.markdown("### 📋 채점 결과")
                    st.markdown(result_text)

                    # 결과 저장
                    file_name = 'results.csv'
                    file_exists = os.path.isfile(file_name)
                    with open(file_name, mode='a', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['선택한문제', '학생답안', '점수', 'AI채점결과'])
                        writer.writerow([selected_q, user_ans, score, result_text.replace('\n', ' ')])
                    st.success("저장 완료!")

# ==================== [탭 2: 학습 분석 & 오답노트] ====================
with tab2:
    st.header("📈 나의 학습 성적표")
    results_file = 'results.csv'
    
    if not os.path.isfile(results_file):
        st.info("💡 아직 저장된 학습 기록이 없습니다. 문제를 풀고 채점해 보세요!")
    else:
        res_df = pd.read_csv(results_file, encoding='utf-8-sig')
        total = len(res_df)
        avg = res_df['점수'].mean() if total > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 풀이 문항", f"{total}개")
        c2.metric("평균 점수", f"{avg:.1f}점")
        c3.metric("학습 상태", "🎯 합격권" if avg >= 60 else "⚠️ 보완 필요")
        
        st.divider()
        st.subheader("📋 전체 학습 기록 및 오답노트")
        st.dataframe(res_df, use_container_width=True)
        
        if st.button("🗑️ 기록 초기화"):
            if os.path.isfile(results_file):
                os.remove(results_file)
                st.rerun()
