"""
아보카도 데이터 심화 분석 연습
데이터 분석 기법들을 단계별로 학습할 수 있는 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
import matplotlib.font_manager as fm
korean_fonts = ['Apple SD Gothic Neo', 'Nanum Gothic', 'AppleGothic', 'Malgun Gothic', 'Dotum']
available_font = None
for font_name in korean_fonts:
    if font_name in [f.name for f in fm.fontManager.ttflist]:
        available_font = font_name
        break
if available_font:
    plt.rcParams['font.family'] = available_font
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data():
    """1단계: 데이터 로딩 및 기본 탐색"""
    print("=== 1단계: 데이터 로딩 및 기본 탐색 ===")
    
    # 데이터 로딩
    df = pd.read_csv('data/avocado.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"데이터 크기: {df.shape}")
    print(f"컬럼: {list(df.columns)}")
    print("\n기본 정보:")
    print(df.info())
    
    print("\n기술통계:")
    print(df.describe())
    
    print("\n결측치 확인:")
    print(df.isnull().sum())
    
    return df

def basic_analysis(df):
    """2단계: 기본 분석"""
    print("\n=== 2단계: 기본 분석 ===")
    
    # 1. 가격 분석
    print("\n1. 가격 분석:")
    print(f"평균 가격: ${df['average_price'].mean():.2f}")
    print(f"최고 가격: ${df['average_price'].max():.2f}")
    print(f"최저 가격: ${df['average_price'].min():.2f}")
    
    # 2. 유형별 분석
    print("\n2. 유형별 분석:")
    type_analysis = df.groupby('type').agg({
        'average_price': ['mean', 'std'],
        'total_volume': 'sum'
    }).round(2)
    print(type_analysis)
    
    # 3. 연도별 트렌드
    print("\n3. 연도별 트렌드:")
    yearly_trend = df.groupby('year').agg({
        'average_price': 'mean',
        'total_volume': 'sum'
    }).round(2)
    print(yearly_trend)
    
    return type_analysis, yearly_trend

def advanced_analysis(df):
    """3단계: 고급 분석"""
    print("\n=== 3단계: 고급 분석 ===")
    
    # 1. 지역별 분석
    print("\n1. 지역별 분석 (상위 10개):")
    region_analysis = df.groupby('geography').agg({
        'average_price': 'mean',
        'total_volume': 'sum'
    }).sort_values('total_volume', ascending=False).head(10)
    print(region_analysis)
    
    # 2. 계절성 분석
    print("\n2. 월별 평균 가격:")
    df['month'] = df['date'].dt.month
    monthly_avg = df.groupby('month')['average_price'].mean().round(2)
    print(monthly_avg)
    
    # 3. 가격 변동성 분석
    print("\n3. 가격 변동성 (표준편차):")
    price_volatility = df.groupby('geography')['average_price'].std().sort_values(ascending=False).head(10)
    print(price_volatility)
    
    return region_analysis, monthly_avg, price_volatility

def create_visualizations(df):
    """4단계: 시각화"""
    print("\n=== 4단계: 시각화 생성 ===")
    
    # 1. 가격 분포 히스토그램
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(df['average_price'], bins=30, alpha=0.7, color='skyblue')
    plt.title('가격 분포')
    plt.xlabel('가격 ($)')
    plt.ylabel('빈도')
    
    # 2. 유형별 가격 박스플롯
    plt.subplot(2, 3, 2)
    df.boxplot(column='average_price', by='type', ax=plt.gca())
    plt.title('유형별 가격 분포')
    plt.suptitle('')  # 기본 제목 제거
    
    # 3. 연도별 평균 가격 트렌드
    plt.subplot(2, 3, 3)
    yearly_price = df.groupby('year')['average_price'].mean()
    yearly_price.plot(kind='line', marker='o')
    plt.title('연도별 평균 가격')
    plt.xlabel('연도')
    plt.ylabel('평균 가격 ($)')
    
    # 4. 월별 가격 패턴
    plt.subplot(2, 3, 4)
    monthly_price = df.groupby('month')['average_price'].mean()
    monthly_price.plot(kind='bar')
    plt.title('월별 평균 가격')
    plt.xlabel('월')
    plt.ylabel('평균 가격 ($)')
    plt.xticks(range(12), ['1월', '2월', '3월', '4월', '5월', '6월', 
                           '7월', '8월', '9월', '10월', '11월', '12월'])
    
    # 5. 가격 vs 거래량 산점도
    plt.subplot(2, 3, 5)
    plt.scatter(df['total_volume'], df['average_price'], alpha=0.5)
    plt.title('거래량 vs 가격')
    plt.xlabel('총 거래량')
    plt.ylabel('평균 가격 ($)')
    
    # 6. 지역별 평균 가격 (상위 10개)
    plt.subplot(2, 3, 6)
    top_regions = df.groupby('geography')['average_price'].mean().sort_values(ascending=False).head(10)
    top_regions.plot(kind='bar')
    plt.title('지역별 평균 가격 (상위 10개)')
    plt.xlabel('지역')
    plt.ylabel('평균 가격 ($)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('avocado_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("시각화가 'avocado_analysis.png'로 저장되었습니다.")

def correlation_analysis(df):
    """5단계: 상관관계 분석"""
    print("\n=== 5단계: 상관관계 분석 ===")
    
    # 수치형 컬럼만 선택
    numeric_cols = ['average_price', 'total_volume', '4046', '4225', '4770', 
                   'total_bags', 'small_bags', 'large_bags', 'xlarge_bags']
    
    correlation_matrix = df[numeric_cols].corr()
    
    print("상관관계 매트릭스:")
    print(correlation_matrix.round(3))
    
    # 상관관계 히트맵
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('변수 간 상관관계')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix

def time_series_analysis(df):
    """6단계: 시계열 분석"""
    print("\n=== 6단계: 시계열 분석 ===")
    
    # 월별 평균 가격 시계열
    monthly_series = df.groupby(df['date'].dt.to_period('M'))['average_price'].mean()
    
    print("월별 평균 가격 (최근 12개월):")
    print(monthly_series.tail(12))
    
    # 시계열 플롯
    plt.figure(figsize=(15, 5))
    monthly_series.plot()
    plt.title('월별 평균 가격 트렌드')
    plt.xlabel('날짜')
    plt.ylabel('평균 가격 ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('time_series.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return monthly_series

def main():
    """메인 실행 함수"""
    print("아보카도 데이터 심화 분석 연습을 시작합니다!")
    print("=" * 50)
    
    # 1단계: 데이터 로딩
    df = load_and_explore_data()
    
    # 2단계: 기본 분석
    type_analysis, yearly_trend = basic_analysis(df)
    
    # 3단계: 고급 분석
    region_analysis, monthly_avg, price_volatility = advanced_analysis(df)
    
    # 4단계: 시각화
    create_visualizations(df)
    
    # 5단계: 상관관계 분석
    correlation_matrix = correlation_analysis(df)
    
    # 6단계: 시계열 분석
    monthly_series = time_series_analysis(df)
    
    print("\n=== 분석 완료! ===")
    print("생성된 파일들:")
    print("- avocado_analysis.png: 기본 시각화")
    print("- correlation_heatmap.png: 상관관계 히트맵")
    print("- time_series.png: 시계열 분석")

if __name__ == "__main__":
    main() 