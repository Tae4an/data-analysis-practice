#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판다스(Pandas) 기초 실습 코드

이 스크립트는 판다스 라이브러리의 핵심 기능들을 단계별로 학습할 수 있도록 구성되었습니다.
- 데이터프레임 생성 및 조작
- 데이터 필터링 및 정렬
- 그룹화 및 집계
- 시각화 기초
- 파일 입출력

Author: 실습생
Date: 2024
"""

# 필요한 라이브러리 임포트
import pandas as pd      # 데이터 분석 및 조작을 위한 핵심 라이브러리
import numpy as np       # 수치 계산 및 배열 처리
import matplotlib.pyplot as plt  # 데이터 시각화
import seaborn as sns    # 고급 통계 시각화 (matplotlib 기반)

# 한글 폰트 설정 (matplotlib에서 한글이 깨지지 않도록 설정)
import matplotlib.font_manager as fm

# macOS에서 사용 가능한 한글 폰트 리스트 (우선순위 순)
korean_fonts = ['Apple SD Gothic Neo', 'Nanum Gothic', 'AppleGothic', 'Malgun Gothic', 'Dotum']

# 사용 가능한 첫 번째 한글 폰트 찾기
available_font = None
for font_name in korean_fonts:
    if font_name in [f.name for f in fm.fontManager.ttflist]:
        available_font = font_name
        break

# 폰트 설정
if available_font:
    plt.rcParams['font.family'] = available_font
    print(f"한글 폰트 설정됨: {available_font}")
else:
    # 대안: DejaVu Sans 사용 (한글은 지원하지 않지만 기본 폰트)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    print("경고: 한글 폰트를 찾을 수 없습니다. 영문 폰트를 사용합니다.")

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# matplotlib 폰트 캐시 초기화 (필요한 경우)
try:
    import matplotlib
    matplotlib.font_manager._rebuild()
except:
    pass

print("="*50)
print("판다스(Pandas) 기초 실습")
print("="*50)

# =================================================================
# 1. 데이터프레임 생성
# =================================================================
print("\n1. 데이터프레임 생성")
print("-" * 30)

# 딕셔너리를 사용해 데이터프레임 생성
# - 각 키는 컬럼명이 되고, 값은 해당 컬럼의 데이터가 됨
# - 모든 리스트의 길이는 동일해야 함 (행의 개수가 같아야 함)
data = {
    '이름': ['김철수', '이영희', '박민수', '정지영', '최동수'],    # 문자열 타입 컬럼
    '나이': [25, 30, 35, 28, 32],                           # 정수 타입 컬럼
    '도시': ['서울', '부산', '대구', '인천', '광주'],          # 문자열 타입 컬럼
    '급여': [3000, 3500, 4000, 3200, 3800],                # 정수 타입 컬럼 (단위: 만원)
    '부서': ['개발', '마케팅', '개발', '인사', '마케팅']        # 범주형 데이터
}

# pandas.DataFrame() 생성자를 사용해 딕셔너리를 데이터프레임으로 변환
df = pd.DataFrame(data)
print("생성된 데이터프레임:")
print(df)
print(f"\n데이터프레임 타입: {type(df)}")  # <class 'pandas.core.frame.DataFrame'>

# =================================================================
# 2. 기본 정보 확인
# =================================================================
print("\n2. 데이터프레임 기본 정보")
print("-" * 30)

# shape: 데이터프레임의 크기를 (행 수, 열 수) 튜플로 반환
print(f"데이터프레임 크기: {df.shape}")  # (5, 5) = 5행 5열

# columns: 컬럼명들을 Index 객체로 반환, list()로 리스트 변환 가능
print(f"컬럼명: {list(df.columns)}")

# dtypes: 각 컬럼의 데이터 타입을 Series로 반환
# - object: 문자열이나 혼합 타입
# - int64: 64비트 정수
# - float64: 64비트 실수
print(f"데이터 타입:\n{df.dtypes}")

# describe(): 수치형 컬럼에 대한 기본 통계 정보 제공
# - count: 결측값이 아닌 데이터 개수
# - mean: 평균값
# - std: 표준편차
# - min, 25%, 50%, 75%, max: 최솟값, 1분위수, 중위수, 3분위수, 최댓값
print(f"\n기본 통계 정보:")
print(df.describe())

# 추가 정보 확인 메서드들
print(f"\n전체 정보 요약:")
print(df.info())  # 메모리 사용량, 결측값 정보 등 포함

# =================================================================
# 3. 데이터 조회 및 선택
# =================================================================
print("\n3. 데이터 조회 및 선택")
print("-" * 30)

# head(n): 상위 n개 행을 반환 (기본값은 5)
# tail(n): 하위 n개 행을 반환
print("첫 3행:")
print(df.head(3))
print("\n마지막 2행:")
print(df.tail(2))

# 단일 컬럼 선택: df['컬럼명'] → Series 반환
# 다중 컬럼 선택: df[['컬럼1', '컬럼2']] → DataFrame 반환
print("\n특정 컬럼 선택 (이름, 급여):")
print(df[['이름', '급여']])

# 단일 컬럼 선택 예시
print("\n단일 컬럼 선택 (이름만):")
names_series = df['이름']  # Series 타입
print(names_series)
print(f"타입: {type(names_series)}")

# 조건부 필터링 (Boolean Indexing)
# 1. 조건식 작성: df['컬럼'] 연산자 값
# 2. 조건식은 True/False로 구성된 Boolean Series를 반환
# 3. 이 Boolean Series를 df[]에 넣으면 True인 행만 필터링됨
print("\n조건부 필터링 (급여 3500 이상):")
condition = df['급여'] >= 3500  # Boolean Series
print(f"조건식 결과: {condition.values}")  # [False True True False True]
high_salary = df[condition]  # 또는 df[df['급여'] >= 3500]
print(high_salary)

# 복합 조건 예시
print("\n복합 조건 (나이 30 이상이면서 급여 3500 이상):")
complex_condition = (df['나이'] >= 30) & (df['급여'] >= 3500)
print(df[complex_condition])

# =================================================================
# 4. 데이터 정렬
# =================================================================
print("\n4. 데이터 정렬")
print("-" * 30)

# sort_values(): 특정 컬럼을 기준으로 정렬
# - by: 정렬 기준 컬럼(들)
# - ascending: True(오름차순, 기본값), False(내림차순)
# - inplace: True면 원본 데이터프레임 수정, False면 새 데이터프레임 반환
print("급여 기준 내림차순 정렬:")
df_sorted = df.sort_values('급여', ascending=False)
print(df_sorted)

print("\n나이 기준 오름차순 정렬:")
df_sorted_age = df.sort_values('나이', ascending=True)
print(df_sorted_age)

# 다중 컬럼 정렬
print("\n부서별 정렬 후 급여 내림차순 정렬:")
df_multi_sorted = df.sort_values(['부서', '급여'], ascending=[True, False])
print(df_multi_sorted)

# sort_index(): 인덱스를 기준으로 정렬
print("\n인덱스 기준 정렬 (내림차순):")
df_index_sorted = df_sorted.sort_index(ascending=False)
print(df_index_sorted)

# =================================================================
# 5. 그룹화 및 집계
# =================================================================
print("\n5. 그룹화 및 집계")
print("-" * 30)

# groupby(): 특정 컬럼의 값에 따라 데이터를 그룹으로 나눔
# - groupby('컬럼명'): 해당 컬럼의 고유값별로 그룹 생성
# - 그룹 객체에 집계 함수를 적용: mean(), sum(), count(), min(), max() 등

print("부서별 평균 급여:")
# 1. '부서'별로 그룹화
# 2. '급여' 컬럼 선택
# 3. mean() 집계 함수 적용
dept_avg_salary = df.groupby('부서')['급여'].mean()
print(dept_avg_salary)
print(f"결과 타입: {type(dept_avg_salary)}")  # pandas.core.series.Series

print("\n부서별 인원 수:")
# size(): 각 그룹의 행 개수 (결측값 포함)
# count(): 각 그룹의 결측값이 아닌 데이터 개수
dept_count = df.groupby('부서').size()
print(dept_count)

# 다양한 집계 함수 예시
print("\n부서별 다양한 통계:")
dept_stats = df.groupby('부서')['급여'].agg(['count', 'mean', 'min', 'max', 'std'])
print(dept_stats)

# 다중 컬럼 집계
print("\n부서별 급여와 나이 통계:")
dept_multi_stats = df.groupby('부서')[['급여', '나이']].mean()
print(dept_multi_stats)

# 그룹별 데이터 확인
print("\n각 부서의 상세 데이터:")
for name, group in df.groupby('부서'):
    print(f"\n{name} 부서:")
    print(group)

# =================================================================
# 6. 새로운 컬럼 추가
# =================================================================
print("\n6. 새로운 컬럼 추가")
print("-" * 30)

# apply() 함수: 각 행이나 열에 함수를 적용
# lambda 함수: 간단한 익명 함수 (lambda 매개변수: 반환값)
print("1) 조건부 컬럼 생성 (경력등급):")
df['경력등급'] = df['나이'].apply(lambda x: '신입' if x < 30 else '경력')
print(df[['이름', '나이', '경력등급']])

# 기존 컬럼을 이용한 계산
print("\n2) 계산 컬럼 생성 (연봉):")
df['연봉'] = df['급여'] * 12  # 월급여 * 12개월
print(df[['이름', '급여', '연봉']])

# 복잡한 조건문을 사용한 컬럼 생성
def salary_grade(salary):
    """급여 등급을 반환하는 함수"""
    if salary >= 4000:
        return '고급'
    elif salary >= 3500:
        return '중급'
    else:
        return '초급'

print("\n3) 복잡한 조건을 사용한 컬럼 생성 (급여등급):")
df['급여등급'] = df['급여'].apply(salary_grade)
print(df[['이름', '급여', '급여등급']])

# 문자열 연산을 이용한 컬럼 생성
print("\n4) 문자열 결합 컬럼 생성 (풀네임):")
df['풀네임'] = df['이름'] + ' (' + df['부서'] + ')'
print(df[['이름', '부서', '풀네임']])

print("\n새로운 컬럼들이 추가된 최종 데이터프레임:")
print(df)

# =================================================================
# 7. 데이터 저장
# =================================================================
print("\n7. 데이터 저장")
print("-" * 30)

# to_csv(): 데이터프레임을 CSV 파일로 저장
# - index=False: 행 인덱스를 파일에 포함하지 않음
# - encoding='utf-8-sig': 한글 깨짐 방지 (Excel에서도 제대로 보임)
# - sep=',': 구분자 설정 (기본값: 쉼표)
df.to_csv('employee_data.csv', index=False, encoding='utf-8-sig')
print("CSV 파일로 저장 완료: employee_data.csv")

# 다른 파일 형식으로 저장 예시
print("\n다른 형식으로도 저장 가능:")
# Excel 파일로 저장 (openpyxl 패키지 필요)
df.to_excel('employee_data.xlsx', index=False, sheet_name='직원정보')
print("Excel 파일로 저장 완료: employee_data.xlsx")

# JSON 파일로 저장
df.to_json('employee_data.json', orient='records', force_ascii=False, indent=2)
print("JSON 파일로 저장 완료: employee_data.json")

# 저장된 파일 다시 읽어오기 예시
print("\n저장된 CSV 파일 다시 읽어오기:")
df_loaded = pd.read_csv('employee_data.csv', encoding='utf-8-sig')
print("읽어온 데이터:")
print(df_loaded.head())

# =================================================================
# 8. 간단한 시각화
# =================================================================
print("\n8. 간단한 시각화")
print("-" * 30)

# matplotlib을 사용한 시각화
# pandas는 matplotlib과 연동되어 쉽게 시각화 가능

# 1) 부서별 평균 급여 막대 그래프
plt.figure(figsize=(10, 6))  # 그래프 크기 설정 (가로 10, 세로 6인치)

# pandas Series의 plot() 메서드 사용
# - kind: 그래프 종류 ('bar', 'line', 'pie', 'hist' 등)
# - color: 색상 설정
dept_avg_salary.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('부서별 평균 급여', fontsize=14, fontweight='bold')  # 제목 설정
plt.xlabel('부서', fontsize=12)  # x축 레이블
plt.ylabel('평균 급여 (만원)', fontsize=12)  # y축 레이블
plt.xticks(rotation=45)  # x축 레이블 45도 회전
plt.grid(axis='y', alpha=0.3)  # y축 격자 표시
plt.tight_layout()  # 레이아웃 자동 조정

# 이미지 파일로 저장
# - dpi: 해상도 (300은 고해상도)
# - bbox_inches='tight': 여백 자동 조정
plt.savefig('department_salary.png', dpi=300, bbox_inches='tight')
plt.show()  # 그래프 화면에 표시

print("시각화 완료: department_salary.png 파일로 저장됨")

# 2) 추가 시각화 예시들
print("\n추가 시각화 예시:")

# 나이 분포 히스토그램
plt.figure(figsize=(8, 5))
df['나이'].plot(kind='hist', bins=5, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('직원 나이 분포')
plt.xlabel('나이')
plt.ylabel('빈도')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 급여 vs 나이 산점도
plt.figure(figsize=(8, 6))
colors = ['red' if dept == '개발' else 'blue' if dept == '마케팅' else 'green' 
          for dept in df['부서']]
plt.scatter(df['나이'], df['급여'], c=colors, s=100, alpha=0.7, edgecolors='black')
plt.title('나이 vs 급여')
plt.xlabel('나이')
plt.ylabel('급여 (만원)')
plt.grid(True, alpha=0.3)

# 범례 추가
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='개발')
blue_patch = mpatches.Patch(color='blue', label='마케팅')
green_patch = mpatches.Patch(color='green', label='인사')
plt.legend(handles=[red_patch, blue_patch, green_patch])

plt.tight_layout()
plt.savefig('age_vs_salary.png', dpi=300, bbox_inches='tight')
plt.show()

print("추가 시각화 완료: age_distribution.png, age_vs_salary.png 파일로 저장됨")

# =================================================================
# 9. 실습용 대용량 데이터 생성
# =================================================================
print("\n9. 실습용 대용량 데이터 생성")
print("-" * 30)

# numpy를 사용한 대용량 랜덤 데이터 생성
# 실제 데이터 분석에서는 보통 수천~수만 행의 데이터를 다루므로 실습용 대용량 데이터 생성

# 재현 가능한 랜덤 데이터를 위한 시드 설정
np.random.seed(42)  # 동일한 랜덤 값이 생성되도록 시드 고정

large_data = {
    '고객ID': range(1, 1001),  # 1부터 1000까지 순차 번호
    # np.random.randint(최솟값, 최댓값, 개수): 정수 랜덤 생성
    '나이': np.random.randint(20, 65, 1000),        # 20~64세 랜덤
    '구매금액': np.random.randint(10000, 500000, 1000),  # 1만~49만원 랜덤
    # np.random.choice(선택지, 개수): 주어진 값들 중 랜덤 선택
    '성별': np.random.choice(['남', '여'], 1000),
    '지역': np.random.choice(['서울', '경기', '부산', '대구', '인천'], 1000),
    '구매횟수': np.random.randint(1, 20, 1000)       # 1~19회 랜덤
}

# 딕셔너리를 데이터프레임으로 변환
large_df = pd.DataFrame(large_data)

# CSV 파일로 저장 (추후 실습용)
large_df.to_csv('customer_data.csv', index=False, encoding='utf-8-sig')

print("대용량 고객 데이터 생성 완료 (1000행)")
print("파일명: customer_data.csv")
print(f"데이터 크기: {large_df.shape}")
print(f"메모리 사용량: {large_df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print("\n미리보기:")
print(large_df.head())

# 생성된 대용량 데이터로 간단한 분석 수행
print("\n=== 대용량 데이터 분석 예시 ===")

print("\n성별별 평균 구매금액:")
gender_avg = large_df.groupby('성별')['구매금액'].mean()
print(gender_avg)
print(f"성별간 구매금액 차이: {gender_avg.max() - gender_avg.min():.0f}원")

print("\n지역별 고객 수:")
region_count = large_df.groupby('지역').size().sort_values(ascending=False)
print(region_count)

print("\n나이대별 구매 패턴:")
# 나이대 구분 (20대, 30대, 40대, 50대, 60대)
large_df['나이대'] = (large_df['나이'] // 10) * 10
age_group_stats = large_df.groupby('나이대')[['구매금액', '구매횟수']].mean()
print(age_group_stats)

print("\n고액 구매고객 분석 (구매금액 상위 10%):")
high_spender_threshold = large_df['구매금액'].quantile(0.9)  # 90분위수
high_spenders = large_df[large_df['구매금액'] >= high_spender_threshold]
print(f"고액 구매 기준: {high_spender_threshold:.0f}원 이상")
print(f"고액 구매고객 수: {len(high_spenders)}명")
print("\n고액 구매고객의 지역 분포:")
print(high_spenders['지역'].value_counts())

# =================================================================
# 실습 완료 및 요약
# =================================================================
print("\n" + "="*50)
print("🎉 판다스 기초 실습 완료! 🎉")
print("="*50)
print("\n📁 생성된 파일:")
print("   📄 employee_data.csv: 직원 데이터 (5행)")
print("   📄 employee_data.xlsx: 직원 데이터 (Excel 형식)")
print("   📄 employee_data.json: 직원 데이터 (JSON 형식)")
print("   📄 customer_data.csv: 고객 데이터 (1000행)")
print("   📊 department_salary.png: 부서별 평균 급여 차트")
print("   📊 age_distribution.png: 나이 분포 히스토그램")
print("   📊 age_vs_salary.png: 나이 vs 급여 산점도")

print("\n🎯 학습한 주요 기능:")
print("   ✅ 데이터프레임 생성 및 기본 정보 확인")
print("   ✅ 데이터 조회, 필터링, 정렬")
print("   ✅ 그룹화 및 집계 분석")
print("   ✅ 새로운 컬럼 생성 및 데이터 변환")
print("   ✅ 파일 입출력 (CSV, Excel, JSON)")
print("   ✅ 기본 시각화")
print("   ✅ 대용량 데이터 생성 및 분석")

print("\n🚀 다음 단계 학습 제안:")
print("   1. 결측값(NaN) 처리 방법")
print("   2. 데이터 병합(merge, join)")
print("   3. 피벗 테이블 생성")
print("   4. 시계열 데이터 분석")
print("   5. 고급 시각화 (seaborn 활용)")

print("="*50) 