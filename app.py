import streamlit as st
import pandas as pd
from openai import OpenAI
import csv
import os
import re
import openpyxl
from io import BytesIO
from PIL import Image as PILImage

# 1. 페이지 설정
st.set_page_config(page_title="건축기사 AI 학습 시스템", layout="wide")
st.title("🏗️ 건축기사 AI 학습 & 채점 시스템")

# 2. Gemini API 키 설정
client = OpenAI(
    api_key=st.secrets["GEMINI_API_KEY"],
    base_url="https://googleapis.com"
)

# 3. 엑셀 데이터 및 이미지 추출 캐싱 함수
@st.cache_data
def load_excel_with_images(file_path):
    try:
        # openpyxl로 워크북 로드
        wb = openpyxl.load_workbook(file_path, data_only=True)
        ws = wb.active
        
        # 1) 데이터프레임으로 기본 텍스트 데이터 변환 (첫 행을 컬럼명으로 사용)
        data = ws.values
        cols = next(data)
        data = list(data)
        df_excel = pd.DataFrame(data, columns=cols)
        
        # 컬럼 공백 제거 및 인덱스 초기화
        df_excel.columns = [str(c).strip() for c in df_excel.columns]
        df_excel = df_excel.dropna(subset=['문제 내용']).reset_index(drop=True)
        
        # 2) 엑셀 내 이미지 추출 및 행 번호와 매핑
        # 행 번호(0부터 시작하는 index)별 바이너리 이미지 데이터를 담을 딕셔너리
        image_dict = {}
        
        if hasattr(ws, '_images'):
            for img in ws._images:
                try:
                    # 엑셀 상의 행/열 인덱스 (0부터 시작)
                    row_idx = img.anchor._from.row - 1 # 헤더 열(1행) 제외하고 df의 index와 맞추기 위함
                    
                    # 이미지 바이트 추출
                    img_bytes = img._data()
                    image_dict[row_idx] = img_bytes
                except Exception:
                    continue
                    
        return df_excel, image_dict
    except FileNotFoundError:
        st.error(f"⚠️ '{file_path}' 파일이 없습니다. 폴더 안에 파일을 먼저 위치시켜 주세요!")
        st.stop()

# 엑셀 파일 로드 실행
df, excel_images = load_excel_with_images('data.xlsx')

# 데이터 전처리 및 새로운 4대 대단원('건축시공', '공정관리', '건축적산', '건축구조') 매핑 함수
df['대단원'] = df['대단원'].astype(str).str.strip()
df['중단원'] = df['중단원'].astype(str).str.strip()
df['문제 내용'] = df['문제 내용'].astype(str).str.strip()

def reclassify_app_units(row):
    text = str(row['문제 내용'])
    old_major = str(row['대단원'])
    old_middle = str(row['중단원'])
    combined = old_major + " " + old_middle + " " + text
    
    # 1) 공정관리
    if any(k in combined for k in ['공정', '네트워크', 'CPM', '공정표', 'VE', '가치공학', '선행작업', '후행작업']):
        return '공정관리'
    # 2) 건축적산
    elif any(k in combined for k in ['적산', '견적', '수량', '단가', '공사비', '물량산출']):
        return '건축적산'
    # 3) 건축구조
    elif any(k in combined for k in ['구조역학', '모멘트', '단면2차', '응력', '보의', '하중', '처짐', '철근콘크리트 구조', '철골구조', '내진', '휨모멘트']):
        return '건축구조'
    # 4) 건축시공 (나머지 전체)
    else:
        return '건축시공'

df['대단원'] = df.apply(reclassify_app_units, axis=1)

# 추출 후 기존 데이터프레임의 인덱스를 보존하기 위한 처리
df = df.reset_index() # 원래 추출된 순서(행 위치)를 'index' 컬럼으로 보존

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
    major_unique = df['대단원'].unique().tolist()
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
        original_row_idx = row_data['index'] # 💡 원래 엑셀의 행 위치 가져오기

        st.info(f"**[출제정보] 연도: {question_year}  |  대단원: {q_major}  |  중단원: {q_sub}**\n\n{selected_q}")
        
        # 💡 [추가] 해당 문제 행에 매칭된 엑셀 내 그림이 존재한다면 문제 밑에 자동 출력
        if original_row_idx in excel_images:
            try:
                img_data = excel_images[original_row_idx]
                image = PILImage.open(BytesIO(img_data))
                st.image(image, caption="[문제 참고 그림]", width=450)
            except Exception as e:
                st.caption("⚠️ 엑셀 내 이미지를 불러오는 과정에서 오류가 발생했습니다.")

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

