import pandas as pd

# 1. 기존 data.csv 불러오기
try:
    df = pd.read_csv('data.csv', encoding='utf-8-sig', engine='python', on_bad_lines='skip')
except UnicodeDecodeError:
    df = pd.read_csv('data.csv', encoding='cp949', engine='python', on_bad_lines='skip')

print(f"정제 전 문제 수: {len(df)}개")

# 2. 강력한 제외 키워드 설정 (그림, 기호, 공정표, 도면, 네트워크 등 포함된 것 전부 컷)
def strict_exclude(text):
    text = str(text)
    keywords = ['공정표', '네트워크', '도면', '스케치', '작도', '그림', '기호']
    for kw in keywords:
        if kw in text:
            return True
    return False

# 3. 필터링 적용
filtered_df = df[~df['문제 내용'].apply(strict_exclude)].copy()

# 4. data.csv로 저장
filtered_df.to_csv('data.csv', index=False, encoding='utf-8-sig')

print(f"정제 완료! 남은 문제 수: {len(filtered_df)}개 (data.csv 저장됨)")
