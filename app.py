import gradio as gr
import pandas as pd
from openai import OpenAI
import csv
import os
import re

# 1. Gemini API 키 설정 (Streamlit secrets 대신 환경변수 또는 직접 입력 지원)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        # Streamlit secrets 구조를 유지하고 있을 경우를 대비한 예외 처리
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 2. CSV 데이터 불러오기
try:
    df = pd.read_csv('data.csv', encoding='cp949')
except FileNotFoundError:
    # cp949 실패 시 utf-8 시도
    try:
        df = pd.read_csv('data.csv', encoding='utf-8')
    except Exception:
        df = pd.DataFrame(columns=['대단원', '중단원', '문제 내용', '모범 답안', '해설'])

major_list = df['대단원'].unique().tolist() if not df.empty else ["기타"]

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

def get_filtered_df(scope_mode, selected_major, target_weak_major):
    if df.empty:
        return df
    if scope_mode == "전체 챕터 랜덤":
        return df
    elif scope_mode == "대단원별 선택":
        return df[df['대단원'] == selected_major]
    else:  # 자주 틀린 취약 대단원 집중 공략
        if target_weak_major:
            return df[df['대단원'] == target_weak_major]
        return df

# ==================== [Gradio UI 구성] ====================
with gr.Blocks(theme=gr.themes.Soft(), title="건축기사 AI 학습 시스템") as demo:
    
    # 상태값(State) 관리 (Gradio State는 꼬이지 않고 안전하게 유지됩니다)
    state_target_weak = gr.State(None)
    
    gr.HTML("<h1>🏗️ 건축기사 AI 학습 & 채점 시스템 (Gradio 버전)</h1>")
    
    with gr.Row():
        # 상단/사이드바 제어 영역
        with gr.Column(scale=1, elem_id="control_panel"):
            gr.Markdown("### 🎛️ 1단계: 학습 범위 설정")
            scope_radio = gr.Radio(
                choices=["전체 챕터 랜덤", "대단원별 선택", "🚨 자주 틀린 취약 대단원 집중 공략"],
                value="전체 챕터 랜덤",
                label="출제 범위 선택"
            )
            
            major_dropdown = gr.Dropdown(
                choices=major_list,
                value=major_list[0] if major_list else None,
                label="대단원 선택 (대단원별 선택 모드 시)",
                visible=False
            )
            
            weak_info_box = gr.Markdown("⚠️ 집중 공략 단원 없음", visible=False)

    # 탭 메뉴 정의 (0: 문제 풀기, 1: 시험지 모드, 2: 오답노트)
    with gr.Tabs() as tabs:
        
        # ----------------- [탭 1: 한 문제씩 풀기 & AI 채팅] -----------------
        with gr.TabItem("🎯 1단계: 문제 풀기 & AI 채팅", id=0):
            gr.Markdown("#### 💡 한 문제씩 집중적으로 풀고 AI의 상세 피드백과 추가 질문을 주고받는 모드입니다.")
            
            with gr.Row():
                q_dropdown = gr.Dropdown(
                    choices=df['문제 내용'].tolist() if not df.empty else [],
                    label="📌 풀고 싶은 문제를 선택하세요:",
                    interactive=True
                )
            
            question_info = gr.Markdown("**출제 단원을 선택해주세요.**")
            user_answer_input = gr.Textbox(label="✍️ 정답을 서술형으로 입력하세요:", lines=4)
            submit_btn = gr.Button("🤖 AI 채점 요청하기", variant="primary")
            
            grading_result = gr.Markdown(label="채점 결과")
            
            gr.Markdown("##### 💬 AI에게 이어서 질문하기")
            chatbot = gr.Chatbot(label="AI 튜터 대화")
            chat_input = gr.Textbox(placeholder="예: 이 개념이 실기 시험에 또 어떻게 응용돼서 나와?", label="추가 질문")
            chat_send_btn = gr.Button("질문 전송")

        # ----------------- [탭 2: 시험지 모드] -----------------
        with gr.TabItem("📑 2단계: 시험지 모드 (일괄 풀이)", id=1):
            gr.Markdown("#### 📑 여러 문제를 시험지처럼 지정한 문항 수만큼 뽑아서 한 번에 풀고 채점하는 모드입니다.")
            
            with gr.Row():
                num_q_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="추출 문항 수")
                refresh_exam_btn = gr.Button("🎲 새로운 문제 세트 무작위 뽑기")
            
            exam_container = gr.Markdown("문항을 불러오는 중...")
            exam_answers_box = gr.Textbox(label="일괄 답안 입력 (줄바꿈 등으로 구분하거나 쉼표 처리)", visible=False)
            batch_submit_btn = gr.Button("📝 전체 답안 일괄 채점 및 저장하기", variant="primary")
            batch_result_output = gr.Markdown()

        # ----------------- [탭 3: 오답노트 & 학습 분석] -----------------
        with gr.TabItem("📊 3단계: 나의 학습 기록 & 오답노트", id=2):
            gr.Markdown("### 📈 나의 학습 성적표 및 취약 챕터 분석")
            stats_output = gr.Markdown("학습 기록을 불러오는 중...")
            weak_analysis_box = gr.Markdown("대단원별 분석을 불러오는 중...")
            
            refresh_stats_btn = gr.Button("🔄 성적표 및 분석 새로고침")
            clear_history_btn = gr.Button("🗑️ 학습 기록 전체 초기화", variant="stop")

    # ==================== [이벤트 연결 및 로직] ====================
    
    # 1. 범위 모드 변경 시 세부 UI 토글 제어
    def update_scope_ui(scope_mode, target_weak):
        if scope_mode == "대단원별 선택":
            return gr.update(visible=True), gr.update(visible=False)
        elif scope_mode == "🚨 자주 틀린 취약 대단원 집중 공략":
            msg = f"🚨 **집중 공략 대단원:** **{target_weak}**" if target_weak else "⚠️ 지정된 취약 대단원이 없습니다. 아래 오답노트에서 [🎯 집중 공략]을 눌러주세요."
            return gr.update(visible=False), gr.update(value=msg, visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)

    scope_radio.change(
        fn=update_scope_ui,
        inputs=[scope_radio, state_target_weak],
        outputs=[major_dropdown, weak_info_box]
    )

    # 2. 범위나 단원이 바뀔 때 문제 목록(Dropdown) 갱신 함수
    def update_question_list(scope_mode, selected_major, target_weak):
        filtered = get_filtered_df(scope_mode, selected_major, target_weak)
        q_list = filtered['문제 내용'].tolist() if not filtered.empty else []
        return gr.update(choices=q_list, value=q_list[0] if q_list else None)

    scope_radio.change(
        fn=update_question_list,
        inputs=[scope_radio, major_dropdown, state_target_weak],
        outputs=[q_dropdown]
    )
    major_dropdown.change(
        fn=update_question_list,
        inputs=[scope_radio, major_dropdown, state_target_weak],
        outputs=[q_dropdown]
    )

    # 3. 문제 선택 시 문제 내용 안내 갱신
    def on_question_select(selected_q):
        if not selected_q or df.empty:
            return "**문제를 선택해주세요.**"
        matched = df[df['문제 내용'] == selected_q]
        if matched.empty:
            return "**문제를 찾을 수 없습니다.**"
        row = matched.iloc[0]
        return f"**[출제단원] 대단원: {row['대단원']}  |  중단원: {row['중단원']}**\n\n{selected_q}"

    q_dropdown.change(
        fn=on_question_select,
        inputs=[q_dropdown],
        outputs=[question_info]
    )

    # 4. AI 채점 요청
    def grade_answer(selected_q, user_ans):
        if not selected_q or not user_ans:
            return "문제와 답안을 모두 입력해주세요!"
        matched = df[df['문제 내용'] == selected_q]
        if matched.empty:
            return "문제를 찾을 수 없습니다."
        row = matched.iloc[0]
        
        prompt = f"""
        너는 건축기사 실기 수석 채점관이야.
        [문제]: {selected_q}
        [모범 답안]: {row['모범 답안']}
        [상세 해설]: {row['해설']}
        [학생 답안]: {user_ans}
        
        핵심 키워드 포함 여부를 엄격히 평가해 0~100점의 점수를 매기고 피드백해줘.
        반드시 아래 형식으로 출력할 것:
        1. 최종 점수: XX점
        2. 키워드 포함 여부: (...)
        3. 채점 상세 평가: (...)
        """
        try:
            response = client.chat.completions.create(
                model="gemini-3.6-flash",
                messages=[{"role": "user", "content": prompt}]
            )
            res_text = response.choices[0].message.content
        except Exception as e:
            res_text = f"채점 중 오류가 발생했습니다: {str(e)}"
            
        score = extract_score(res_text)
        
        # CSV 저장
        file_name = 'results.csv'
        file_exists = os.path.isfile(file_name)
        with open(file_name, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['선택한문제', '학생답안', '점수', 'AI채점결과'])
            writer.writerow([selected_q, user_ans, score, res_text.replace('\n', ' ')])
            
        return res_text

    submit_btn.click(
        fn=grade_answer,
        inputs=[q_dropdown, user_answer_input],
        outputs=[grading_result]
    )

    # 5. 오답노트 & 학습 통계 생성 함수
    def load_learning_stats():
        results_file = 'results.csv'
        if not os.path.isfile(results_file):
            return "💡 아직 저장된 학습 기록이 없습니다. 문제를 풀고 채점해 보세요!", "분석할 데이터가 없습니다."
        
        res_df = pd.read_csv(results_file, encoding='utf-8-sig')
        total = len(res_df)
        avg = res_df['점수'].mean() if total > 0 else 0
        status = "🎯 합격권" if avg >= 60 else "⚠️ 보완 필요"
        
        summary_md = f"""
        - **총 풀이 문항:** {total}개
        - **평균 점수:** {avg:.1f}점
        - **학습 상태:** {status}
        """
        
        res_df['대단원'], res_df['중단원'] = zip(*res_df['선택한문제'].apply(lambda x: find_chapter_info(x, df)))
        major_stats = res_df.groupby('대단원').agg(
            평균점수=('점수', 'mean'),
            풀이횟수=('점수', 'count')
        ).reset_index().sort_values(by='평균점수', ascending=True)
        
        weak_md = "#### 🚨 대단원별 취약 과목 분석\n"
        for _, row in major_stats.iterrows():
            weak_md += f"- 📂 **[{row['대단원']}]** (풀이: {row['풀이횟수']}회, 평균 점수: **{row['평균점수']:.1f}점**)\n"
            
        return summary_md, weak_md

    demo.load(fn=load_learning_stats, outputs=[stats_output, weak_analysis_box])
    refresh_stats_btn.click(fn=load_learning_stats, outputs=[stats_output, weak_analysis_box])

    # 6. 🔥 핵심 요청: 오답노트에서 '집중 공략' 버튼 클릭 시 단원 지정 + 모드 변경 + 탭 0(문제 풀기)로 강제 이동!
    def trigger_weak_target(major_name):
        # 반환 순서: state_target_weak, scope_radio, major_dropdown, weak_info_box, q_dropdown, tabs (탭 인덱스 0으로 이동)
        q_list = df[df['대단원'] == major_name]['문제 내용'].tolist()
        new_q_dropdown = gr.update(choices=q_list, value=q_list[0] if q_list else None)
        weak_msg = f"🚨 **집중 공략 대단원:** **{major_name}**"
        
        return (
            major_name, 
            "🚨 자주 틀린 취약 대단원 집중 공략", 
            gr.update(visible=False), 
            gr.update(value=weak_msg, visible=True),
            new_q_dropdown,
            0 # tabs 컴포넌트의 selected를 0번 탭(문제 풀기)으로 전환
        )

    # 7. 전체 기록 초기화
    def clear_history():
        results_file = 'results.csv'
        if os.path.isfile(results_file):
            os.remove(results_file)
        return "학습 기록이 초기화되었습니다.", "데이터가 없습니다."

    clear_history_btn.click(fn=clear_history, outputs=[stats_output, weak_analysis_box])

# ==================== [앱 실행] ====================
if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
