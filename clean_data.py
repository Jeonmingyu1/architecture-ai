import pandas as pd

# 1. 기존 data.csv 파일 불러오기
try:
    df = pd.read_csv('data.csv', encoding='utf-8-sig', engine='python', on_bad_lines='skip')
except UnicodeDecodeError:
    df = pd.read_csv('data.csv', encoding='cp949', engine='python', on_bad_lines='skip')

print(f"기존 전체 문제 수: {len(df)}개")

# 2. 그림, 도면, 공정표, 스케치 등 텍스트 서술이 곤란한 문제 키워드 필터링
def should_exclude(text):
    text = str(text)
    # 제외할 키워드 목록
    patterns = [
        '공정표', '네트워크', '도면', '스케치', '작도', 
        '그림과 같은', '그림을', '기호로 표기', '표에'
    ]
    for p in patterns:
        if p in text:
            return True
    return False

# 3. 필터링 적용 (조건에 안 맞는 순수 텍스트/계산 문제만 남김)
filtered_df = df[~df['문제 내용'].apply(should_exclude)].copy()

# 4. data.csv 파일로 덮어쓰기 저장
filtered_df.to_csv('data.csv', index=False, encoding='utf-8-sig')

print(f"정제 완료! 제외된 문제 수: {len(df) - len(filtered_df)}개")
print(f"최종 남은 문제 수: {len(filtered_df)}개 (data.csv 저장 완료)")