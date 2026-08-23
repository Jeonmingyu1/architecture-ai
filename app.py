import streamlit as st
import pandas as pd
from openai import OpenAI
import csv
import os
import re

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

# --- 헬퍼 함수 ---
def find_chapter_info(q_text, full_df):
    matched = full_df[full_df['문제 내용'] == q_text]
    if not matched.empty:
        return matched.iloc[0]['대단원'], matched.iloc[0]['중단원']
    return "기타", "기타"

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

# ==================== [세션 상태 초기화] ====================
if 'target_weak_major' not in st.session_state:
    st.session_state['target_weak_major'] = None
if 'scope_mode' not in st.session_state:
    st.session_state['scope_mode'] = "전체 챕터 랜덤"

# ==================== [상단 제어 바: 학습 범위 설정 콜백] ====================
def on_scope_change():
    st.session_state['scope_mode'] = st.session_state['scope_selector_key']
    if st.session_state['scope_mode'] != "🚨 자주 틀린 취약 대단원 집중 공략":
        st.session_state['target_weak_major'] = None
    if 'batch_exam_df' in st.session_state:
        del st.session_state['batch_exam_df']

st.markdown("### 🎛️ 1단계: 학습 범위 설정")

scope_options = ["전체 챕터 랜덤", "대단원별 선택", "🚨 자주 틀린 취약 대단원 집중 공략"]

if st.session_state['scope_mode'] not in scope_options:
    st.session_state['scope_mode'] = "전체 챕터 랜덤"

c_scope, c_detail = st.columns([1.5, 2.5])

with c_scope:
    st.selectbox(
        "출제 범위 선택", 
        scope_options, 
        index=scope_options.index(st.session_state['scope_mode']), 
        key="scope_selector_key",
        on_change=on_scope_change
    )

target_df = pd.DataFrame()
major_list = df['대단원'].unique().tolist()

if st.session_state['scope_mode'] == "전체 챕터 랜덤":
    with c_detail:
        st.markdown("<br><b>전체 데이터베이스 대상 (모든 챕터 포함)</b>", unsafe_allow_html=True)
    target_df = df

elif st.session_state['scope_mode'] == "대단원별 선택":
    with c_detail:
        def on_major_change():
            st.session_state['selected_major_val'] = st.session_state['major_selector_key']
            if 'batch_exam_df' in st.session_state:
                del st.session_state['batch_exam_df']

        prev_major = st.session_state.get('selected_major_val', major_list[0])
        if prev_major not in major_list:
            prev_major = major_list[0]
            
        selected_major = st.selectbox(
            "대단원 선택", 
            major_list, 
            index=major_list.index(prev_major), 
            key="major_selector_key",
            on_change=on_major_change
        )
    # 💡 오타 수정 완료 (= 를 == 로 변경)
    current_major = st.session_state.get('selected_major_val', major_list[0])
    target_df = df[df['대단원'] == current_major]

else:  # 자주 틀린 취약 대단원 집중 공략
    weak_major = st.session_state['target_weak_major']
    with c_detail:
        if weak_major:
            st.markdown(f"<br>🚨 집중 공략 대단원: <b>{weak_major}</b>", unsafe_allow_html=True)
            target_df = df[df['대단원'] == weak_major]
        else:
            st.markdown("<br>⚠️ <b>지정된 취약 대단원 없음 (3단계 오답노트에서 '집중 공략'을 눌러주세요)</b>", unsafe_allow_html=True)
            target_df = pd.DataFrame()

st.divider()

# ==================== [메인 대메뉴 탭] ====================
main_tab1, main_tab2, main_tab3 = st.tabs(["🎯 2단계: 문제 풀기 & AI 채팅", "📑 3단계: 시험지 모드 (일괄 풀이)", "📊 나의 학습 기록 & 오답노트"])

# ==================== [탭 1: 한 문제씩 풀기 + 이어서 질문하기] ====================
with main_tab1:
    st.markdown("#### 💡 한 문제씩 집중적으로 풀고 AI의 상세 피드백과 추가 질문을 주고받는 모드입니다.")
    
    q_list = target_df['문제 내용'].tolist() if not target_df.empty else []
    if not q_list:
        st.warning("선택된 범위에 문제가 없거나 집중 공략할 대단원이 지정되지 않았습니다. 3단계(오답노트)에서 '집중 공략' 버튼을 눌러주세요.")
    else:
        selected_q = st.selectbox("📌 풀고 싶은 문제를 선택하세요:", q_list, key="single_q_select")
        row_data = target_df[target_df['문제 내용'] == selected_q].iloc[0]
        
        correct_answer = row_data['모범 답안']
        explanation = row_data['해설']

        st.info(f"**[출제단원] 대단원: {row_data['대단원']}  |  중단원: {row_data['중단원']}**\n\n{selected_q}")
        user_ans = st.text_area("✍️ 정답을 서술형으로 입력하세요:", height=120, key="single_user_ans")

        if st.button("🤖 AI 채점 요청하기", type="primary"):
            if not user_ans:
                st.warning("답안을 입력해주세요!")
            else:
                with st.spinner("AI 채점 중..."):
                    prompt = f"""
                    너는 건축기사 실기 수석 채점관이야.
                    [문제]: {selected_q}
                    [모범 답안]: {correct_answer}
                    [상세 해설]: {explanation}
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

                    st.session_state['messages'] = [
                        {"role": "user", "content": f"문제: {selected_q}\n내 답안: {user_ans}"},
                        {"role": "assistant", "content": result_text}
                    ]

                    file_name = 'results.csv'
                    file_exists = os.path.isfile(file_name)
                    with open(file_name, mode='a', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['선택한문제', '학생답안', '점수', 'AI채점결과'])
                        writer.writerow([selected_q, user_ans, score, result_text.replace('\n', ' ')])
                    st.success("채점 완료 및 오답노트 저장 완료!")

        # --- 이어서 질문하기 (채팅 영역) ---
        st.divider()
        st.markdown("##### 💬 AI에게 이어서 질문하기")
        st.caption("채점 결과에 대해 궁금한 점이나 더 알고 싶은 개념을 자유롭게 물어보세요!")

        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        for message in st.session_state['messages']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if chat_input := st.chat_input("예: 이 개념이 실기 시험에 또 어떻게 응용돼서 나와?"):
            st.session_state['messages'].append({"role": "user", "content": chat_input})
            with st.chat_message("user"):
                st.markdown(chat_input)

            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    chat_history = [
                        {
                            "role": "system", 
                            "content": f"너는 건축기사 수석 강사야. 현재 풀고 있는 문제는 '{selected_q}'이고 모범 답안은 '{correct_answer}', 해설은 '{explanation}'이야. 학생의 추가 질문에 친절하고 전문적으로 답해줘."
                        }
                    ] + st.session_state['messages']

                    chat_response = client.chat.completions.create(
                        model="gemini-3.6-flash",
                        messages=chat_history
                    )
                    answer_text = chat_response.choices[0].message.content
                    st.markdown(answer_text)
                    st.session_state['messages'].append({"role": "assistant", "content": answer_text})

# ==================== [탭 2: 시험지 모드] ====================
with main_tab2:
    st.markdown("#### 📑 여러 문제를 시험지처럼 지정한 문항 수만큼 뽑아서 한 번에 풀고 채점하는 모드입니다.")
    
    if target_df.empty:
        st.warning("선택된 범위에 문제가 없습니다. (취약 챕터 모드인 경우 3단계 오답노트에서 '집중 공략' 버튼을 먼저 눌러주세요)")
    else:
        c_cnt, c_action = st.columns([2, 2])
        with c_cnt:
            max_limit = len(target_df)
            num_q = st.number_input("추출 문항 수 설정", min_value=1, max_value=max(1, max_limit), value=min(5, max_limit), key="batch_num_q")
        with c_action:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎲 새로운 문제 세트 무작위 뽑기", use_container_width=True, type="secondary"):
                st.session_state['batch_exam_df'] = target_df.sample(n=num_q).reset_index(drop=True)
                st.rerun()

        if 'batch_exam_df' not in st.session_state or len(st.session_state['batch_exam_df']) != num_q:
            st.session_state['batch_exam_df'] = target_df.sample(n=num_q).reset_index(drop=True)

        exam_df = st.session_state['batch_exam_df']
        st.divider()

        user_answers_dict = {}
        for idx, row in exam_df.iterrows():
            st.markdown(f"**Q{idx+1}. [{row['대단원']} > {row['중단원']}] {row['문제 내용']}**")
            ans = st.text_area(f"답안 입력 (문항 {idx+1})", key=f"batch_ans_{idx}", height=90)
            user_answers_dict[idx] = {
                "question": row['문제 내용'],
                "correct": row['모범 답안'],
                "explanation": row['해설'],
                "user_ans": ans
            }
            st.markdown("---")

        if st.button("📝 전체 답안 일괄 채점 및 저장하기", type="primary", use_container_width=True):
            with st.spinner("🤖 AI가 전체 답안을 채점 중입니다..."):
                file_name = 'results.csv'
                file_exists = os.path.isfile(file_name)
                batch_results = []

                for idx, data in user_answers_dict.items():
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
                    
                    with open(file_name, mode='a', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['선택한문제', '학생답안', '점수', 'AI채점결과'])
                            file_exists = True
                        writer.writerow([data['question'], data['user_ans'], score, res_text.replace('\n', ' ')])

                st.success("🎉 일괄 채점이 완료되었습니다! 결과를 확인하세요.")
                for res in batch_results:
                    with st.expander(f"📌 [점수: {res['score']}점] {res['question'][:35]}..."):
                        st.markdown(f"**내 답안:** {res['user_ans']}")
                        st.markdown(f"**AI 채점 결과:**\n{res['result']}")

# ==================== [탭 3: 학습 분석 & 오답노트] ====================
with main_tab3:
    st.header("📈 나의 학습 성적표 및 취약 챕터 분석")
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
        
        # --- 대단원별 취약 챕터 분석 ---
        st.subheader("🚨 대단원별 자주 틀리는 취약 과목 분석")
        
        res_df['대단원'], res_df['중단원'] = zip(*res_df['선택한문제'].apply(lambda x: find_chapter_info(x, df)))
        
        major_stats = res_df.groupby('대단원').agg(
            평균점수=('점수', 'mean'),
            풀이횟수=('점수', 'count')
        ).reset_index()
        
        weak_majors = major_stats.sort_values(by='평균점수', ascending=True)
        
        if not weak_majors.empty:
            st.markdown("👇 대단원별 **취약 과목** 분석입니다. 버튼을 누르면 해당 대단원만 모아서 즉시 집중 학습할 수 있습니다!")
            
            for idx, row in weak_majors.head(5).iterrows():
                major_name = row['대단원']
                avg_s = row['평균점수']
                count = row['풀이횟수']
                
                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    st.markdown(f"- 📂 **대단원: [{major_name}]** (풀이: {count}회, 평균 점수: **{avg_s:.1f}점**)")
                with col_btn:
                    if st.button(f"🎯 집중 공략", key=f"weak_btn_fixed_{idx}"):
                        st.session_state['target_weak_major'] = major_name
                        st.session_state['scope_mode'] = "🚨 자주 틀린 취약 대단원 집중 공략"
                        if 'batch_exam_df' in st.session_state:
                            del st.session_state['batch_exam_df']
                        st.rerun()
        
        st.divider()
        st.subheader("📋 전체 학습 기록 데이터")
        st.dataframe(res_df[['선택한문제', '대단원', '중단원', '학생답안', '점수', 'AI채점결과']], use_container_width=True)
        
        if st.button("🗑️ 학습 기록 전체 초기화"):
            if os.path.isfile(results_file):
                os.remove(results_file)
                st.rerun()
