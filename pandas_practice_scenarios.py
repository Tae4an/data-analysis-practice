#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판다스 연습 시나리오 모음

이 파일은 판다스를 단계별로 연습할 수 있는 다양한 시나리오들을 제공합니다.
각 시나리오는 독립적으로 실행 가능하며, 난이도별로 구성되었습니다.

사용법:
1. 전체 실행: python pandas_practice_scenarios.py
2. 특정 시나리오만 실행: 함수를 개별 호출
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # 경고 메시지 숨기기

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

def scenario_1_basic_operations():
    """
    🟢 시나리오 1: 기본 조작 연습 (초급)
    - 데이터프레임 생성
    - 기본 정보 확인
    - 간단한 필터링
    """
    print("="*60)
    print("🟢 시나리오 1: 기본 조작 연습")
    print("="*60)
    
    # 학생 성적 데이터 생성
    students = {
        '이름': ['김민수', '이지은', '박철수', '정유진', '최준호'],
        '수학': [85, 92, 78, 96, 88],
        '영어': [90, 85, 82, 94, 76],
        '과학': [88, 90, 85, 92, 80],
        '학년': [1, 2, 1, 3, 2]
    }
    
    df = pd.DataFrame(students)
    
    print("📚 학생 성적 데이터:")
    print(df)
    
    # 연습 문제들
    print("\n📝 연습 문제:")
    print("1. 데이터프레임의 크기는?")
    print(f"   답: {df.shape}")
    
    print("\n2. 수학 점수가 85점 이상인 학생은?")
    high_math = df[df['수학'] >= 85]
    print(high_math[['이름', '수학']])
    
    print("\n3. 각 과목의 평균 점수는?")
    subjects = ['수학', '영어', '과학']
    for subject in subjects:
        avg = df[subject].mean()
        print(f"   {subject}: {avg:.1f}점")
    
    print("\n4. 총점과 평균 추가하기:")
    df['총점'] = df['수학'] + df['영어'] + df['과학']
    df['평균'] = df['총점'] / 3
    print(df[['이름', '총점', '평균']])
    
    return df

def scenario_2_data_analysis():
    """
    🟡 시나리오 2: 데이터 분석 연습 (중급)
    - 그룹화 및 집계
    - 조건부 분석
    - 데이터 변환
    """
    print("\n" + "="*60)
    print("🟡 시나리오 2: 매장 매출 데이터 분석")
    print("="*60)
    
    # 매장 매출 데이터 생성
    np.random.seed(123)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    sales_data = {
        '날짜': np.random.choice(dates, 200),
        '매장': np.random.choice(['강남점', '홍대점', '건대점', '신촌점'], 200),
        '상품카테고리': np.random.choice(['전자제품', '의류', '식품', '도서'], 200),
        '매출액': np.random.randint(50000, 500000, 200),
        '고객수': np.random.randint(10, 100, 200)
    }
    
    df = pd.DataFrame(sales_data)
    df = df.sort_values('날짜').reset_index(drop=True)
    
    print("🏪 매장 매출 데이터 (200건):")
    print(df.head(10))
    
    print("\n📊 분석 결과:")
    
    print("\n1. 매장별 총 매출액:")
    store_sales = df.groupby('매장')['매출액'].sum().sort_values(ascending=False)
    print(store_sales)
    
    print("\n2. 상품카테고리별 평균 매출액:")
    category_avg = df.groupby('상품카테고리')['매출액'].mean().sort_values(ascending=False)
    print(category_avg)
    
    print("\n3. 매장별 고객당 평균 구매액:")
    df['고객당구매액'] = df['매출액'] / df['고객수']
    store_avg_per_customer = df.groupby('매장')['고객당구매액'].mean()
    print(store_avg_per_customer)
    
    print("\n4. 고매출 거래 분석 (상위 20%):")
    high_sales_threshold = df['매출액'].quantile(0.8)
    high_sales = df[df['매출액'] >= high_sales_threshold]
    print(f"   고매출 기준: {high_sales_threshold:,.0f}원 이상")
    print("   고매출 거래의 매장별 분포:")
    print(high_sales['매장'].value_counts())
    
    return df

def scenario_3_time_series():
    """
    🔴 시나리오 3: 시계열 데이터 분석 (고급)
    - 날짜 데이터 처리
    - 시계열 분석
    - 트렌드 분석
    """
    print("\n" + "="*60)
    print("🔴 시나리오 3: 주식 가격 시계열 분석")
    print("="*60)
    
    # 가상의 주식 가격 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    
    # 주식 가격 시뮬레이션 (랜덤 워크)
    initial_price = 100000
    returns = np.random.normal(0.001, 0.02, 90)  # 일 수익률 (평균 0.1%, 변동성 2%)
    prices = [initial_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    stock_data = {
        '날짜': dates,
        '종가': prices,
        '거래량': np.random.randint(100000, 1000000, 90)
    }
    
    df = pd.DataFrame(stock_data)
    df['날짜'] = pd.to_datetime(df['날짜'])  # 날짜 타입으로 변환
    df = df.set_index('날짜')  # 날짜를 인덱스로 설정
    
    print("📈 주식 가격 데이터:")
    print(df.head(10))
    
    # 시계열 분석
    print("\n📊 시계열 분석:")
    
    print("\n1. 기본 통계:")
    print(f"   기간: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"   최고가: {df['종가'].max():,.0f}원")
    print(f"   최저가: {df['종가'].min():,.0f}원")
    print(f"   평균가: {df['종가'].mean():,.0f}원")
    
    print("\n2. 월별 평균 주가:")
    monthly_avg = df['종가'].resample('M').mean()
    for date, price in monthly_avg.items():
        print(f"   {date.strftime('%Y년 %m월')}: {price:,.0f}원")
    
    print("\n3. 이동평균선 계산:")
    df['MA5'] = df['종가'].rolling(window=5).mean()    # 5일 이동평균
    df['MA20'] = df['종가'].rolling(window=20).mean()  # 20일 이동평균
    
    print("   최근 5일 데이터:")
    print(df[['종가', 'MA5', 'MA20']].tail())
    
    print("\n4. 변동성 분석:")
    df['일수익률'] = df['종가'].pct_change()  # 일간 수익률
    volatility = df['일수익률'].std() * np.sqrt(252)  # 연환산 변동성
    print(f"   일간 평균 변동률: {df['일수익률'].mean()*100:.3f}%")
    print(f"   연환산 변동성: {volatility*100:.1f}%")
    
    return df

def scenario_4_data_cleaning():
    """
    🟠 시나리오 4: 데이터 정제 연습 (실무형)
    - 결측값 처리
    - 이상값 탐지
    - 데이터 변환
    """
    print("\n" + "="*60)
    print("🟠 시나리오 4: 고객 데이터 정제")
    print("="*60)
    
    # 문제가 있는 데이터 생성 (실제 상황 시뮬레이션)
    np.random.seed(456)
    
    dirty_data = {
        '고객ID': range(1, 101),
        '이름': [f'고객{i}' if i % 10 != 0 else None for i in range(1, 101)],  # 10%는 결측값
        '나이': [np.random.randint(20, 70) if np.random.random() > 0.05 
                else None for _ in range(100)],  # 5%는 결측값
        '소득': [np.random.randint(2000, 8000) * 10000 if np.random.random() > 0.1
                else np.random.choice([None, 99999999]) for _ in range(100)],  # 결측값과 이상값
        '이메일': [f'user{i}@email.com' if i % 15 != 0 else 'invalid_email' 
                  for i in range(1, 101)],  # 일부는 잘못된 형식
        '가입일': pd.date_range('2020-01-01', periods=100, freq='3D')
    }
    
    df = pd.DataFrame(dirty_data)
    
    print("🧹 정제 전 데이터 상태:")
    print(f"데이터 크기: {df.shape}")
    print("\n결측값 현황:")
    print(df.isnull().sum())
    
    print("\n이상값 탐지:")
    print(f"소득 통계: 최솟값={df['소득'].min():,}, 최댓값={df['소득'].max():,}")
    
    # 데이터 정제 과정
    print("\n🔧 데이터 정제 과정:")
    
    # 1. 결측값 처리
    print("\n1. 결측값 처리:")
    df['이름'] = df['이름'].fillna('미등록')  # 결측값을 '미등록'으로 대체
    df['나이'] = df['나이'].fillna(df['나이'].median())  # 결측값을 중위수로 대체
    print("   - 이름: '미등록'으로 대체")
    print("   - 나이: 중위수로 대체")
    
    # 2. 이상값 처리
    print("\n2. 이상값 처리:")
    # 소득이 1억 이상인 것을 이상값으로 간주
    outliers = df['소득'] > 100000000
    df.loc[outliers, '소득'] = df['소득'].median()
    df['소득'] = df['소득'].fillna(df['소득'].median())
    print("   - 소득: 1억 이상을 중위수로 대체")
    
    # 3. 이메일 형식 검증 및 수정
    print("\n3. 이메일 형식 수정:")
    invalid_emails = ~df['이메일'].str.contains('@', na=False)
    df.loc[invalid_emails, '이메일'] = df.loc[invalid_emails, '이메일'] + '@corrected.com'
    print("   - 잘못된 이메일 형식 수정")
    
    print("\n🎯 정제 후 데이터 상태:")
    print(f"결측값: {df.isnull().sum().sum()}개")
    print(f"소득 범위: {df['소득'].min():,} ~ {df['소득'].max():,}")
    
    print("\n정제된 데이터 샘플:")
    print(df.head(10))
    
    return df

def scenario_5_visualization():
    """
    🟣 시나리오 5: 데이터 시각화 연습
    - 다양한 차트 유형
    - 서브플롯
    - 스타일링
    """
    print("\n" + "="*60)
    print("🟣 시나리오 5: 종합 데이터 시각화")
    print("="*60)
    
    # 종합 분석용 데이터 생성
    np.random.seed(789)
    
    # 카페 매출 데이터
    months = ['1월', '2월', '3월', '4월', '5월', '6월']
    menu_items = ['아메리카노', '라떼', '프라푸치노', '케이크', '샌드위치']
    
    cafe_data = []
    for month in months:
        for item in menu_items:
            sales = np.random.randint(50, 200)
            price = {'아메리카노': 4500, '라떼': 5000, '프라푸치노': 6000, 
                    '케이크': 7000, '샌드위치': 8000}[item]
            cafe_data.append({
                '월': month,
                '상품': item,
                '판매량': sales,
                '단가': price,
                '매출': sales * price
            })
    
    df = pd.DataFrame(cafe_data)
    
    print("☕ 카페 매출 데이터:")
    print(df.head(10))
    
    # 다양한 시각화 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('카페 매출 종합 분석', fontsize=16, fontweight='bold')
    
    # 1. 월별 총 매출 (선 그래프)
    monthly_sales = df.groupby('월')['매출'].sum()
    axes[0, 0].plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, color='blue')
    axes[0, 0].set_title('월별 총 매출 추이')
    axes[0, 0].set_ylabel('매출 (원)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 상품별 평균 판매량 (막대 그래프)
    item_avg_sales = df.groupby('상품')['판매량'].mean()
    axes[0, 1].bar(item_avg_sales.index, item_avg_sales.values, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('상품별 평균 판매량')
    axes[0, 1].set_ylabel('판매량')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 상품별 매출 비율 (파이 차트)
    item_total_sales = df.groupby('상품')['매출'].sum()
    axes[1, 0].pie(item_total_sales.values, labels=item_total_sales.index, autopct='%1.1f%%')
    axes[1, 0].set_title('상품별 매출 비율')
    
    # 4. 월별 상품 판매량 히트맵
    pivot_data = df.pivot_table(values='판매량', index='상품', columns='월', aggfunc='mean')
    im = axes[1, 1].imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_xticks(range(len(pivot_data.columns)))
    axes[1, 1].set_yticks(range(len(pivot_data.index)))
    axes[1, 1].set_xticklabels(pivot_data.columns)
    axes[1, 1].set_yticklabels(pivot_data.index)
    axes[1, 1].set_title('월별 상품 판매량 히트맵')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('cafe_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n📊 시각화 저장 완료: cafe_analysis.png")
    
    return df

def run_all_scenarios():
    """모든 시나리오를 순차적으로 실행"""
    print("🚀 판다스 연습 시나리오 전체 실행")
    print("="*60)
    
    # 각 시나리오 실행
    df1 = scenario_1_basic_operations()
    df2 = scenario_2_data_analysis()
    df3 = scenario_3_time_series()
    df4 = scenario_4_data_cleaning()
    df5 = scenario_5_visualization()
    
    print("\n" + "="*60)
    print("🎉 모든 시나리오 완료!")
    print("="*60)
    print("\n📁 생성된 파일:")
    print("   📊 cafe_analysis.png: 카페 매출 종합 분석")
    
    print("\n🎯 연습한 기능들:")
    print("   ✅ 기본 데이터프레임 조작")
    print("   ✅ 그룹화 및 집계 분석")
    print("   ✅ 시계열 데이터 처리")
    print("   ✅ 데이터 정제 (결측값, 이상값)")
    print("   ✅ 종합 데이터 시각화")
    
    return {
        'students': df1,
        'sales': df2,
        'stocks': df3,
        'customers': df4,
        'cafe': df5
    }

if __name__ == "__main__":
    # 전체 시나리오 실행
    results = run_all_scenarios()
    
    print("\n🔥 추가 연습 과제:")
    print("1. 학생 데이터에서 학년별 성적 분석하기")
    print("2. 매장 데이터에서 요일별 매출 패턴 찾기")
    print("3. 주식 데이터에서 골든크로스 신호 찾기")
    print("4. 고객 데이터에서 나이대별 소득 분석하기")
    print("5. 카페 데이터에서 시즌별 메뉴 성과 분석하기")
    
    print("\n💡 개별 시나리오 실행 방법:")
    print("   scenario_1_basic_operations()")
    print("   scenario_2_data_analysis()")
    print("   scenario_3_time_series()")
    print("   scenario_4_data_cleaning()")
    print("   scenario_5_visualization()") 