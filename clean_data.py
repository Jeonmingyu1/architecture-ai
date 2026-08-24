import pandas as pd

# 1. 인코딩 에러 방지하며 data.csv 읽기
try:
    df = pd.read_csv('data.csv', encoding='utf-8-sig', engine='python', on_bad_lines='skip')
except:
    df = pd.read_csv('data.csv', encoding='cp949', engine='python', on_bad_lines='skip')

print(f"정제 전 총 문제 수: {len(df)}")

# 2. 그림, 기호, 공정표, 도면 등이 들어간 문제 내용 필터링 조건
keywords = ['공정표', '네트워크', '도면', '스케치', '작도', '그림', '기호']

def is_excluded(text):
    text = str(text)
    for kw in keywords:
        if kw in text:
            return True
    return False

# 3. 제외 대상이 아닌 순수 문제들만 추출
clean_df = df[~df['문제 내용'].apply(is_excluded)].copy()

# 4. 기존 data.csv 파일을 완전히 덮어쓰기 저장
clean_df.to_csv('data.csv', index=False, encoding='utf-8-sig')

print(f"정제 후 남은 문제 수: {len(clean_df)}")
print("data.csv 파일이 성공적으로 갱신되었습니다!")
