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

# 3. Excel 데이터 불러오기 및 전처리
try:
    raw_df = pd.read_excel('data.xlsx', engine='openpyxl')
except FileNotFoundError:
    st.error("⚠️ 'data.xlsx' 파일이 없습니다. 폴더 안에 data.xlsx 파일을 먼저 위치시켜 주세요!")
    st.stop()

# 컬럼명 공백 제거
raw_df.columns = raw_df.columns.str.strip()

def clean_val(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() == 'nan':
        return ""
    return s

# 💡 [핵심] '문제 내용' 열의 밑에 있는 셀에 그림이 들어오는 구조를 반영한 전처리 루프
processed_rows = []
i = 0
while i < len(raw_df):
    row = raw_df.iloc[i]
    q_text = clean_val(row.get('문제 내용'))
    major = clean_val(row.get('대단원'))
    
    # 완전히 빈 행은 스킵
    if major == "" and q_text == "":
        i += 1
        continue
        
    middle = clean_val(row.get('중단원'))
    year = clean_val(row.get('년도'))
    correct = clean_val(row.get('모범 답안'))
    explanation = clean_val(row.get('해설'))
    
    img_path = None
    
    # 바로 다음 행(i + 1)이 존재할 때, '문제 내용' 열 바로 아래 셀에 이미지가 있는지 확인
    if i + 1 < len(raw_df):
        next_row = raw_df.iloc[i + 1]
        next_q_text = clean_val(next_row.get('문제 내용'))
        next_major = clean_val(next_row.get('대단원'))
        next_middle = clean_val(next_row.get('중단원'))
        
        # 조건: 대단원/중단원 등 다른 열은 비어있고, 오직 '문제 내용' 열의 밑에만 파일 확장자나 경로가 적혀있는 경우
        if next_major == "" and next_middle == "" and \
           any(ext in next_q_text.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', 'images/']):
            img_path = next_q_text
            i += 1  # 이미지 행을 소모했으므로 인덱스 추가 증가
            
    processed_rows.append({
        '대단원': major,
        '중단원': middle,
        '년도': year,
        '문제 내용': q_text,
        '모범 답안': correct,
        '해설': explanation,
        '이미지': img_path
    })
    i += 1

df = pd.DataFrame(processed_rows)

# 4대 대단원 자동 재분류 적용
def reclassify_app_units(row):
    text = str(row['문제 내용'])
    old_major = str(row['대단원'])
    old_middle = str(row['중단원'])
    combined = old_major + " " + old_middle + " " + text
    
    if any(k in combined for k in ['공정', '네트워크', 'CPM', '공정표', 'VE', '가치공학', '선행작업', '후행작업']):
        return '공정관리'
    elif any(k in combined for k in ['적산', '견적', '수량', '단가', '공사비', '물량산출']):
        return '건축적산'
    elif any(k in combined for k in ['구조역학', '모멘트', '단면2차', '응력', '보의', '하중', '처짐', '철근콘크리트 구조', '철골구조', '내진', '휨모멘트']):
        return '건축구조'
    else:
        return '건축시공'

df['대단원'] = df.apply(reclassify_app_units, axis=1)

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
if 'scope_mode' not in st.session_state:
    st.session_state['scope_mode'] = "🎲 전체 챕터"
if 'target_weak_major' not in st.session_state:
    st.session_state['target_weak_major'] = None
if 'selected_major_val' not in st.session_state:
    major_unique = df['대단원'].unique().tolist() if not df.empty else []
    st.session_state['selected_major_val'] = major_unique[0] if major_unique else ""
if 'active_tab_index' not in st.session_state:
    st.session_state['active_tab_index'] = 0

# ==================== [사이드바: 학습 범위 설정] ====================
st.sidebar.markdown("### 🎛️ 공부할 범위 고르기")

current_mode = st.session_state['scope_mode']
st.sidebar.markdown(f"현재 학습 모드: **{current_mode}**")

if st.sidebar.button("🎲 전체 챕터로 변경", use_container_width=True):
    st.session_state['scope_mode'] = "🎲 전체 챕터"
    st.session_state['target_weak_major'] = None
    if 'batch_exam_df' in st.session_state:
        del st.session_state['batch_exam_df']
    st.rerun()

major_list = df['대단원'].unique().tolist() if not df.empty else []
selected_major_sb = st.sidebar.selectbox("📚 챕터별 학습 (대단원 선택)", major_list)
if st.sidebar.button("📚 선택한 챕터로 공부 시작", use_container_width=True):
    st.session_state['scope_mode'] = "📚 챕터별 학습"
    st.session_state['selected_major_val'] = selected_major_sb
    st.session_state['target_weak_major'] = None
    if 'batch_exam_df' in st.session_state:
        del st.session_state['batch_exam_df']
    st.rerun()

if st.session_state['scope_mode'] == "🚨 취약 파트 공부" and st.session_state['target_weak_major']:
    st.sidebar.warning(f"🚨 **집중 공략 중인 파트:**\n\n**{st.session_state['target_weak_major']}**")

st.sidebar.divider()

# --- 대상 데이터프레임 필터링 로직 ---
target_df = pd.DataFrame()
if st.session_state['scope_mode'] == "🎲 전체 챕터":
    target_df = df
elif st.session_state['scope_mode'] == "📚 챕터별 학습":
    target_df = df[df['대단원'] == st.session_state['selected_major_val']]
elif st.session_state['scope_mode'] == "🚨 취약 파트 공부":
    weak_m = st.session_state['target_weak_major']
    if weak_m:
        target_df = df[df['대단원'] == weak_m]
    else:
        target_df = df

# ==================== [메인 상단 네비게이션 버튼] ====================
col_t1, col_t2, col_t3 = st.columns(3)

with col_t1:
    if st.button("🎯 1단계: 문제 풀기 & AI", use_container_width=True, type="primary" if st.session_state['active_tab_index']==0 else "secondary"):
        st.session_state['active_tab_index'] = 0
        st.rerun()
with col_t2:
    if st.button("📑 2단계: 시험지 모드", use_container_width=True, type="primary" if st.session_state['active_tab_index']==1 else "secondary"):
        st.session_state['active_tab_index'] = 1
        st.rerun()
with col_t3:
    if st.button("📊 3단계: 성적표 & 오답", use_container_width=True, type="primary" if st.session_state['active_tab_index']==2 else "secondary"):
        st.session_state['active_tab_index'] = 2
        st.rerun()

st.divider()

# 🖼️ 이미지 출력 헬퍼 함수
def render_question_image(row_data):
    img_path = row_data.get('이미지')
    if img_path and str(img_path).strip() != "":
        path_str = str(img_path).strip()
        if os.path.exists(path_str):
            st.image(path_str, caption="[문제 참고 그림]", use_column_width=True)
        else:
            try:
                st.image(path_str, caption="[문제 참고 그림]", use_column_width=True)
            except Exception:
                pass

# ==================== [탭 1: 한 문제씩 풀기 + 이어서 질문하기] ====================
if st.session_state['active_tab_index'] == 0:
    if st.session_state['scope_mode'] == "🚨 취약 파트 공부":
        st.info(f"🚨 현재 **[{st.session_state['target_weak_major']}]** 파트 집중 공략 모드입니다!")
    else:
        st.markdown("#### 💡 한 문제씩 집중적으로 풀고 AI의 채점 결과와 모범 답안을 즉시 확인하는 모드입니다.")
    
    q_list = target_df['문제 내용'].tolist() if not target_df.empty else []
    if not q_list:
        st.warning("⚠️ 선택된 범위에 문제가 없습니다. 사이드바나 3단계 성적표에서 파트를 다시 선택해 주세요.")
    else:
        selected_q = st.selectbox("📌 풀고 싶은 문제를 선택하세요:", q_list, key="single_q_select")
        row_data = target_df[target_df['문제 내용'] == selected_q].iloc[0]
        
        correct_answer = row_data['모범 답안']
        explanation = row_data['해설']
        question_year = row_data.get('년도', '정보 없음')
        q_major = row_data['대단원']
        q_sub = row_data['중단원']

        st.info(f"**[출제정보] 연도: {question_year}  |  대단원: {q_major}  |  중단원: {q_sub}**\n\n{selected_q}")
        
        # 그림 출력
        render_question_image(row_data)

        user_ans = st.text_area("✍️ 정답을 서술형으로 입력하세요:", height=120, key="single_user_ans")

        if st.button("🤖 AI 채점 요청하기", type="primary"):
            if not user_ans:
                st.warning("답안을 입력해주세요!")
            else:
                with st.spinner("AI 채점 중..."):
                    prompt = f"""
                    너는 건축기사 실기 수석 채점관이야.
                    [출제 연도]: {question_year}
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

                    st.session_state['last_graded'] = {
                        "question": selected_q,
                        "user_ans": user_ans,
                        "score": score,
                        "result_text": result_text,
                        "correct_answer": correct_answer,
                        "explanation": explanation
                    }

                    st.session_state['messages'] = [
                        {"role": "user", "content": f"문제: {selected_q}\n내 답안: {user_ans}"},
                        {"role": "assistant", "content": result_text}
                    ]

                    file_name = 'results.csv'
                    file_exists = os.path.isfile(file_name)
                    with open(file_name, mode='a', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['선택한문제', '대단원', '중단원', '년도', '학생답안', '점수', 'AI채점결과'])
                        writer.writerow([selected_q, q_major, q_sub, question_year, user_ans, score, result_text.replace('\n', ' ')])
                    st.success("채점 완료 및 오답노트 저장 완료!")

        if 'last_graded' in st.session_state and st.session_state['last_graded']['question'] == selected_q:
            lg = st.session_state['last_graded']
            st.markdown("---")
            st.markdown("### 📋 채점 결과 및 정답 확인")
            st.info(f"**점수: {lg['score']}점**")
            st.markdown(lg['result_text'])
            
            st.success(f"**📖 모범 답안**\n\n{lg['correct_answer']}")
            st.info(f"**💡 상세 해설**\n\n{lg['explanation']}")
            st.markdown("---")

        st.markdown("##### 💬 AI에게 이어서 질문하기")
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
                            "content": f"너는 건축기사 수석 강사야. 현재 풀고 있는 문제는 '{selected_q}' (출제연도: {question_year})이고 모범 답안은 '{correct_answer}', 해설은 '{explanation}'이야. 학생의 추가 질문에 친절하고 전문적으로 답해줘."
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
elif st.session_state['active_tab_index'] == 1:
    st.markdown("#### 📑 여러 문제를 시험지처럼 지정한 문항 수만큼 뽑아서 한 번에 풀고 채점하는 모드입니다.")
    
    if target_df.empty:
        st.warning("⚠️ 선택된 범위에 문제가 없습니다.")
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
            q_year = row.get('년도', '정보 없음')
            st.markdown(f"**Q{idx+1}. [{q_year} | {row['대단원']} > {row['중단원']}] {row['문제 내용']}**")
            
            # 그림 출력
            render_question_image(row)

            ans = st.text_area(f"답안 입력 (문항 {idx+1})", key=f"batch_ans_{idx}", height=90)
            user_answers_dict[idx] = {
                "question": row['문제 내용'],
                "major": row['대단원'],
                "sub": row['중단원'],
                "year": q_year,
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
                        
                    batch_results.append({
                        "question": data['question'], 
                        "user_ans": data['user_ans'], 
                        "score": score, 
                        "result": res_text,
                        "correct": data['correct'],
                        "explanation": data['explanation']
                    })
                    
                    with open(file_name, mode='a', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['선택한문제', '대단원', '중단원', '년도', '학생답안', '점수', 'AI채점결과'])
                            file_exists = True
                        writer.writerow([data['question'], data['major'], data['sub'], data['year'], data['user_ans'], score, res_text.replace('\n', ' ')])

                st.success("🎉 일괄 채점이 완료되었습니다!")
                for res in batch_results:
                    with st.expander(f"📌 [점수: {res['score']}점] {res['question'][:35]}..."):
                        st.markdown(f"**내 답안:** {res['user_ans']}")
                        st.markdown(f"**AI 채점 결과:**\n{res['result']}")
                        st.markdown("---")
                        st.markdown(f"**[모범 답안]**\n{res['correct']}")
                        st.markdown(f"**[상세 해설]**\n{res['explanation']}")

# ==================== [탭 3: 학습 분석 & 오답노트] ====================
elif st.session_state['active_tab_index'] == 2:
    st.header("📈 나의 학습 성적표 및 취약 챕터 분석")
    results_file = 'results.csv'
    
    if not os.path.isfile(results_file):
        st.info("💡 아직 저장된 학습 기록이 없습니다. 문제를 풀고 채점해 보세요!")
    else:
        res_df = pd.read_csv(results_file, encoding='utf-8-sig')
        if '대단원' not in res_df.columns:
            res_df['대단원'], res_df['중단원'], res_df['년도'] = zip(*res_df['선택한문제'].apply(lambda x: (df[df['문제 내용'] == x].iloc[0]['대단원'] if not df[df['문제 내용'] == x].empty else '건축시공', '기타', '기타')))
        else:
            res_df['대단원'] = res_df['대단원'].apply(lambda m: m if m in ['건축시공', '공정관리', '건축적산', '건축구조'] else '건축시공')

        total = len(res_df)
        avg = res_df['점수'].mean() if total > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 풀이 문항", f"{total}개")
        c2.metric("평균 점수", f"{avg:.1f}점")
        c3.metric("학습 상태", "🎯 합격권" if avg >= 60 else "⚠️ 보완 필요")
        
        st.divider()
        
        st.subheader("🚨 파트별 성적 분석 및 취약 파트 공부 추천")
        major_stats = res_df.groupby('대단원').agg(
            평균점수=('점수', 'mean'),
            풀이횟수=('점수', 'count')
        ).reset_index()
        
        weak_majors = major_stats.sort_values(by='평균점수', ascending=True)
        
        if not weak_majors.empty:
            st.markdown("👇 점수가 낮게 나온 파트의 **[🎯 집중 공략]** 버튼을 누르면, 곧바로 해당 파트의 문제만 풀 수 있도록 1단계 화면으로 이동합니다!")
            
            for idx, row in weak_majors.head(5).iterrows():
                major_name = row['대단원']
                avg_s = row['평균점수']
                count = row['풀이횟수']
                
                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    st.markdown(f"- 📂 **파트: [{major_name}]** (풀이: {count}회, 평균 점수: **{avg_s:.1f}점**)")
                with col_btn:
                    if st.button(f"🎯 집중 공략", key=f"fixed_focus_btn_{idx}", type="primary"):
                        st.session_state['target_weak_major'] = major_name
                        st.session_state['scope_mode'] = "🚨 취약 파트 공부"
                        st.session_state['active_tab_index'] = 0 
                        if 'batch_exam_df' in st.session_state:
                            del st.session_state['batch_exam_df']
                        st.rerun()
        
        st.divider()
        st.subheader("📋 전체 학습 기록 데이터")
        st.dataframe(res_df[['선택한문제', '대단원', '중단원', '년도', '학생답안', '점수', 'AI채Test결과' if 'AI채Test결과' in res_df.columns else 'AI채점결과']], use_column_width=True)
        
        if st.button("🗑️ 학습 기록 전체 초기화"):
            if os.path.isfile(results_file):
                os.remove(results_file)
                st.rerun()
