#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판다스 데이터 시각화 대시보드

실제 CSV 데이터를 활용한 Interactive 데이터 분석 도구
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="데이터 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# 데이터 로딩 함수
@st.cache_data
def load_employee_data():
    """직원 데이터 로딩"""
    try:
        df = pd.read_csv('employee_data.csv')
        return df
    except FileNotFoundError:
        # CSV 파일이 없으면 샘플 데이터 생성
        data = {
            '이름': ['김철수', '이영희', '박민수', '정지영', '최동수'],
            '나이': [25, 30, 35, 28, 32],
            '도시': ['서울', '부산', '대구', '인천', '광주'],
            '급여': [3000, 3500, 4000, 3200, 3800],
            '부서': ['개발', '마케팅', '개발', '인사', '마케팅']
        }
        return pd.DataFrame(data)

@st.cache_data
def load_sales_data():
    """매출 데이터 로딩"""
    try:
        df = pd.read_csv('sales_data.csv')
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['객단가'] = df['매출액'] / df['고객수']
        return df
    except FileNotFoundError:
        # CSV 파일이 없으면 샘플 데이터 생성
        np.random.seed(123)
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        
        data = {
            '날짜': np.random.choice(dates, 50),
            '매장': np.random.choice(['강남점', '홍대점', '건대점', '신촌점'], 50),
            '상품': np.random.choice(['커피', '음료', '디저트', '샐러드'], 50),
            '매출액': np.random.randint(20000, 80000, 50),
            '고객수': np.random.randint(8, 25, 50)
        }
        
        df = pd.DataFrame(data)
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['객단가'] = df['매출액'] / df['고객수']
        return df

@st.cache_data
def load_avocado_data():
    """아보카도 시장 데이터 로딩"""
    try:
        df = pd.read_csv('data/avocado.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error("아보카도 데이터 파일을 찾을 수 없습니다.")
        return None

# 제목
st.title("데이터 분석 대시보드")
st.markdown("판다스와 Streamlit을 활용한 Interactive 분석 도구")
st.markdown("---")

# 사이드바
st.sidebar.title("분석 메뉴")
page = st.sidebar.selectbox(
    "분석할 데이터를 선택하세요:",
    ["홈", "직원 데이터 분석", "매출 데이터 분석", "아보카도 시장 분석", "데이터 업로드"]
)

# 홈 페이지
if page == "홈":
    st.header("환영합니다")
    
    # 상단 소개 컨테이너
    with st.container():    
        intro_col1, intro_col2, intro_col3 = st.columns([1, 2, 1])
        with intro_col2:
            st.markdown("""
            <div style='text-align: center; padding: 20px; background-color: #747474; border-radius: 10px; margin: 20px 0;'>
                <h3>데이터 분석의 모든 것을 한 곳에서</h3>
                <p>실시간 데이터 시각화부터 상세 분석까지, 직관적인 대시보드로 경험해보세요</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 메인 기능 소개 (4열 레이아웃)
    st.subheader("주요 분석 기능")
    feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
    
    with feature_col1:
        with st.container():
            st.markdown("**직원 데이터 분석**")
            st.write("• 부서별 급여 분석")
            st.write("• 나이와 급여 상관관계")
            st.write("• 도시별 분포 현황")
            st.write("• Interactive 필터링")
    
    with feature_col2:
        with st.container():
            st.markdown("**매출 데이터 분석**")
            st.write("• 매장별 매출 비교")
            st.write("• 시계열 매출 추이")
            st.write("• 상품별 성과 분석")
            st.write("• 실시간 KPI 모니터링")
    
    with feature_col3:
        with st.container():
            st.markdown("**아보카도 시장 분석**")
            st.write("• 가격 트렌드 분석")
            st.write("• 지역별 시장 비교")
            st.write("• 유기농 vs 일반 분석")
            st.write("• 품종별 상세 분석")
    
    with feature_col4:
        with st.container():
            st.markdown("**데이터 업로드**")
            st.write("• CSV 파일 업로드")
            st.write("• 자동 데이터 탐지")
            st.write("• 실시간 시각화")
            st.write("• 결과 다운로드")
    
    # 기술 스택 (확장 가능한 섹션)
    with st.expander("사용된 기술 스택 보기"):
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
        
        with tech_col1:
            st.markdown("**프론트엔드**")
            st.write("- Streamlit")
            st.write("- HTML/CSS")
        
        with tech_col2:
            st.markdown("**데이터 처리**")
            st.write("- Pandas")
            st.write("- NumPy")
        
        with tech_col3:
            st.markdown("**시각화**")
            st.write("- Plotly")
            st.write("- Matplotlib")
        
        with tech_col4:
            st.markdown("**기타**")
            st.write("- Python 3.13")
            st.write("- Seaborn")
    
    # 시작하기 버튼 (중앙 정렬)
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
    with button_col2:
        st.info("왼쪽 사이드바에서 분석할 데이터를 선택해주세요!")

# 직원 데이터 분석
elif page == "직원 데이터 분석":
    st.header("직원 데이터 분석")
    
    # 데이터 로딩
    df = load_employee_data()
    
    # 필터링 옵션 (사이드바)
    st.sidebar.subheader("필터 옵션")
    selected_dept = st.sidebar.multiselect("부서 선택:", df['부서'].unique(), default=df['부서'].unique())
    age_range = st.sidebar.slider("나이 범위:", int(df['나이'].min()), int(df['나이'].max()), 
                                 (int(df['나이'].min()), int(df['나이'].max())))
    
    # 필터링된 데이터
    filtered_df = df[
        (df['부서'].isin(selected_dept)) & 
        (df['나이'] >= age_range[0]) & 
        (df['나이'] <= age_range[1])
    ]
    
    # KPI 메트릭 카드 (5열 레이아웃)
    with st.container():
        st.subheader("핵심 지표")
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
        
        with metric_col1:
            st.metric("총 직원 수", len(filtered_df), delta=f"{len(filtered_df) - len(df)} (필터적용)")
        with metric_col2:
            st.metric("평균 급여", f"{filtered_df['급여'].mean():.0f}만원")
        with metric_col3:
            st.metric("최고 급여", f"{filtered_df['급여'].max():.0f}만원")
        with metric_col4:
            st.metric("평균 나이", f"{filtered_df['나이'].mean():.1f}세")
        with metric_col5:
            st.metric("활성 부서", f"{filtered_df['부서'].nunique()}개")
    
    # 탭으로 구분된 분석 섹션
    tab1, tab2, tab3, tab4 = st.tabs(["급여 분석", "부서 분석", "지역 분석", "원본 데이터"])
    
    with tab1:
        st.subheader("급여 관련 분석")
        
        # 급여 분석 (2x2 레이아웃)
        salary_col1, salary_col2 = st.columns(2)
        
        with salary_col1:
            with st.container():
                st.markdown("**부서별 평균 급여**")
                dept_salary = filtered_df.groupby('부서')['급여'].mean().reset_index()
                fig = px.bar(dept_salary, x='부서', y='급여', 
                           title="부서별 평균 급여 비교",
                           color='급여', color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
        
        with salary_col2:
            with st.container():
                st.markdown("**나이와 급여 관계**")
                fig = px.scatter(filtered_df, x='나이', y='급여', color='부서',
                               title="나이와 급여의 상관관계",
                               size_max=20, opacity=0.7)
                st.plotly_chart(fig, use_container_width=True)
        
        # 급여 분포 (전체 너비)
        with st.container():
            st.markdown("**급여 분포 상세 분석**")
            sal_detail_col1, sal_detail_col2, sal_detail_col3 = st.columns(3)
            
            with sal_detail_col1:
                fig = px.histogram(filtered_df, x='급여', nbins=8,
                                 title="급여 분포")
                st.plotly_chart(fig, use_container_width=True)
            
            with sal_detail_col2:
                fig = px.box(filtered_df, x='부서', y='급여',
                           title="부서별 급여 박스플롯")
                st.plotly_chart(fig, use_container_width=True)
            
            with sal_detail_col3:
                # 급여 구간별 통계
                st.markdown("**급여 구간별 인원**")
                salary_ranges = pd.cut(filtered_df['급여'], bins=3, labels=['하위', '중위', '상위'])
                range_counts = salary_ranges.value_counts().reset_index()
                range_counts.columns = ['급여구간', '인원수']
                fig = px.pie(range_counts, values='인원수', names='급여구간',
                           title="급여 구간별 분포")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("부서별 상세 분석")
        
        dept_main_col1, dept_main_col2 = st.columns([2, 1])
        
        with dept_main_col1:
            # 부서별 상세 통계표
            dept_stats = filtered_df.groupby('부서').agg({
                '급여': ['mean', 'max', 'min', 'count'],
                '나이': 'mean'
            }).round(1)
            dept_stats.columns = ['평균급여', '최고급여', '최저급여', '인원수', '평균나이']
            st.dataframe(dept_stats, use_container_width=True)
        
        with dept_main_col2:
            # 부서별 인원 분포
            dept_count = filtered_df['부서'].value_counts().reset_index()
            dept_count.columns = ['부서', '인원수']
            fig = px.pie(dept_count, values='인원수', names='부서',
                       title="부서별 인원 분포")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("지역별 분석")
        
        region_col1, region_col2, region_col3 = st.columns(3)
        
        with region_col1:
            city_count = filtered_df['도시'].value_counts().reset_index()
            city_count.columns = ['도시', '직원수']
            fig = px.bar(city_count, x='도시', y='직원수',
                       title="도시별 직원 수")
            st.plotly_chart(fig, use_container_width=True)
        
        with region_col2:
            city_salary = filtered_df.groupby('도시')['급여'].mean().reset_index()
            fig = px.bar(city_salary, x='도시', y='급여',
                       title="도시별 평균 급여")
            st.plotly_chart(fig, use_container_width=True)
        
        with region_col3:
            # 도시-부서 매트릭스
            city_dept = pd.crosstab(filtered_df['도시'], filtered_df['부서'])
            fig = px.imshow(city_dept.values,
                          x=city_dept.columns,
                          y=city_dept.index,
                          title="도시-부서 분포 히트맵",
                          color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("원본 데이터")
        
        # 데이터 표시 옵션
        data_col1, data_col2, data_col3 = st.columns([1, 1, 2])
        
        with data_col1:
            show_all = st.checkbox("전체 데이터 보기")
        with data_col2:
            sort_column = st.selectbox("정렬 기준:", filtered_df.columns)
        
        # 데이터 표시
        if show_all:
            display_df = filtered_df.sort_values(sort_column, ascending=False)
        else:
            display_df = filtered_df.sort_values(sort_column, ascending=False).head(10)
        
        st.dataframe(display_df, use_container_width=True)
        
        # 다운로드 버튼
        csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="필터링된 데이터 다운로드",
            data=csv,
            file_name='filtered_employee_data.csv',
            mime='text/csv'
        )

# 매출 데이터 분석
elif page == "매출 데이터 분석":
    st.header("매출 데이터 분석")
    
    # 데이터 로딩
    df = load_sales_data()
    
    # 사이드바 필터
    st.sidebar.subheader("분석 필터")
    selected_stores = st.sidebar.multiselect("매장 선택:", df['매장'].unique(), default=df['매장'].unique())
    selected_products = st.sidebar.multiselect("상품 선택:", df['상품'].unique(), default=df['상품'].unique())
    
    # 날짜 범위 필터
    date_range = st.sidebar.date_input(
        "분석 기간:",
        value=[df['날짜'].min(), df['날짜'].max()],
        min_value=df['날짜'].min(),
        max_value=df['날짜'].max()
    )
    
    # 필터링 적용
    filtered_df = df[
        (df['매장'].isin(selected_stores)) & 
        (df['상품'].isin(selected_products)) &
        (df['날짜'] >= pd.to_datetime(date_range[0])) &
        (df['날짜'] <= pd.to_datetime(date_range[1]))
    ]
    
    # 대시보드 KPI 섹션 (6열 레이아웃)
    with st.container():
        st.subheader("매출 대시보드")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
        
        with kpi_col1:
            total_sales = filtered_df['매출액'].sum()
            st.metric("총 매출", f"{total_sales:,}원")
        with kpi_col2:
            total_customers = filtered_df['고객수'].sum()
            st.metric("총 고객수", f"{total_customers:,}명")
        with kpi_col3:
            avg_order = filtered_df['객단가'].mean()
            st.metric("평균 객단가", f"{avg_order:.0f}원")
        with kpi_col4:
            transaction_count = len(filtered_df)
            st.metric("거래 건수", f"{transaction_count:,}건")
        with kpi_col5:
            best_store = filtered_df.groupby('매장')['매출액'].sum().idxmax()
            st.metric("최고 매장", best_store)
        with kpi_col6:
            best_product = filtered_df.groupby('상품')['매출액'].sum().idxmax()
            st.metric("인기 상품", best_product)
    
    # 분석 탭 구성
    analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
        "매장 분석", "상품 분석", "트렌드 분석", "상세 데이터"
    ])
    
    with analysis_tab1:
        st.subheader("매장별 성과 분석")
        
        # 매장 분석 메인 컨테이너
        with st.container():
            store_main_col1, store_main_col2 = st.columns([3, 2])
            
            with store_main_col1:
                # 매장별 상세 통계
                store_stats = filtered_df.groupby('매장').agg({
                    '매출액': ['sum', 'mean', 'count'],
                    '고객수': 'sum',
                    '객단가': 'mean'
                }).round(0)
                store_stats.columns = ['총매출', '평균매출', '거래수', '총고객', '평균객단가']
                st.dataframe(store_stats, use_container_width=True)
            
            with store_main_col2:
                # 매장별 매출 파이차트
                store_sales = filtered_df.groupby('매장')['매출액'].sum().reset_index()
                fig = px.pie(store_sales, values='매출액', names='매장',
                           title="매장별 매출 비율")
                st.plotly_chart(fig, use_container_width=True)
        
        # 매장 상세 차트 (3열 레이아웃)
        with st.container():
            store_detail_col1, store_detail_col2, store_detail_col3 = st.columns(3)
            
            with store_detail_col1:
                fig = px.bar(store_sales, x='매장', y='매출액',
                           title="매장별 총 매출",
                           color='매출액', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            
            with store_detail_col2:
                store_customers = filtered_df.groupby('매장')['고객수'].sum().reset_index()
                fig = px.bar(store_customers, x='매장', y='고객수',
                           title="매장별 총 고객수",
                           color='고객수', color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
            
            with store_detail_col3:
                store_avg_order = filtered_df.groupby('매장')['객단가'].mean().reset_index()
                fig = px.bar(store_avg_order, x='매장', y='객단가',
                           title="매장별 평균 객단가",
                           color='객단가', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
    
    with analysis_tab2:
        st.subheader("상품별 성과 분석")
        
        # 상품 분석 레이아웃
        product_col1, product_col2 = st.columns([2, 3])
        
        with product_col1:
            # 상품별 통계
            product_stats = filtered_df.groupby('상품').agg({
                '매출액': ['sum', 'mean'],
                '고객수': 'sum',
                '객단가': 'mean'
            }).round(0)
            product_stats.columns = ['총매출', '평균매출', '총고객', '평균객단가']
            st.dataframe(product_stats, use_container_width=True)
            
            # 상품별 매출 순위
            product_ranking = filtered_df.groupby('상품')['매출액'].sum().sort_values(ascending=False).reset_index()
            product_ranking['순위'] = range(1, len(product_ranking) + 1)
            st.dataframe(product_ranking[['순위', '상품', '매출액']], use_container_width=True)
        
        with product_col2:
            # 상품-매장 매트릭스
            product_store_matrix = filtered_df.groupby(['상품', '매장'])['매출액'].sum().unstack(fill_value=0)
            fig = px.imshow(product_store_matrix.values,
                          x=product_store_matrix.columns,
                          y=product_store_matrix.index,
                          title="상품별 매장 성과 히트맵",
                          color_continuous_scale='YlOrRd')
            st.plotly_chart(fig, use_container_width=True)
        
        # 상품 상세 차트
        with st.container():
            product_chart_col1, product_chart_col2 = st.columns(2)
            
            with product_chart_col1:
                product_sales = filtered_df.groupby('상품')['매출액'].sum().reset_index()
                fig = px.treemap(product_sales, path=['상품'], values='매출액',
                               title="상품별 매출 트리맵")
                st.plotly_chart(fig, use_container_width=True)
            
            with product_chart_col2:
                fig = px.sunburst(filtered_df, path=['상품', '매장'], values='매출액',
                                title="상품-매장 계층 분석")
                st.plotly_chart(fig, use_container_width=True)
    
    with analysis_tab3:
        st.subheader("시계열 트렌드 분석")
        
        # 트렌드 분석 메인 차트
        with st.container():
            daily_sales = filtered_df.groupby('날짜')['매출액'].sum().reset_index()
            fig = px.line(daily_sales, x='날짜', y='매출액',
                         title="일별 매출 추이",
                         markers=True, line_shape='spline')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # 트렌드 상세 분석 (2열)
        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            # 매장별 일별 매출
            daily_store_sales = filtered_df.groupby(['날짜', '매장'])['매출액'].sum().reset_index()
            fig = px.line(daily_store_sales, x='날짜', y='매출액', color='매장',
                         title="매장별 일별 매출 추이")
            st.plotly_chart(fig, use_container_width=True)
        
        with trend_col2:
            # 상품별 일별 매출
            daily_product_sales = filtered_df.groupby(['날짜', '상품'])['매출액'].sum().reset_index()
            fig = px.bar(daily_product_sales, x='날짜', y='매출액', color='상품',
                        title="상품별 일별 매출 구성")
            st.plotly_chart(fig, use_container_width=True)
        
        # 성장률 분석
        with st.expander("성장률 분석 보기"):
            growth_col1, growth_col2, growth_col3 = st.columns(3)
            
            with growth_col1:
                if len(daily_sales) > 1:
                    growth_rate = ((daily_sales['매출액'].iloc[-1] - daily_sales['매출액'].iloc[0]) / daily_sales['매출액'].iloc[0] * 100)
                    st.metric("전체 성장률", f"{growth_rate:.1f}%")
            
            with growth_col2:
                avg_daily_sales = daily_sales['매출액'].mean()
                st.metric("일평균 매출", f"{avg_daily_sales:,.0f}원")
            
            with growth_col3:
                max_daily_sales = daily_sales['매출액'].max()
                max_date = daily_sales.loc[daily_sales['매출액'].idxmax(), '날짜']
                st.metric("최고 매출일", f"{max_date.strftime('%m-%d')}", f"{max_daily_sales:,.0f}원")
    
    with analysis_tab4:
        st.subheader("상세 거래 데이터")
        
        # 데이터 컨트롤 패널
        with st.container():
            data_control_col1, data_control_col2, data_control_col3, data_control_col4 = st.columns(4)
            
            with data_control_col1:
                show_rows = st.selectbox("표시할 행 수:", [10, 20, 50, 100])
            with data_control_col2:
                sort_by = st.selectbox("정렬 기준:", ['날짜', '매출액', '고객수', '객단가'])
            with data_control_col3:
                sort_order = st.radio("정렬 순서:", ['내림차순', '오름차순'])
            with data_control_col4:
                st.write("") # 공간 확보
                search_term = st.text_input("매장/상품 검색:")
        
        # 데이터 필터링 및 정렬
        display_df = filtered_df.copy()
        
        if search_term:
            mask = (display_df['매장'].str.contains(search_term, case=False, na=False) | 
                   display_df['상품'].str.contains(search_term, case=False, na=False))
            display_df = display_df[mask]
        
        ascending = True if sort_order == '오름차순' else False
        display_df = display_df.sort_values(sort_by, ascending=ascending).head(show_rows)
        
        # 데이터 표시
        st.dataframe(display_df, use_container_width=True)
        
        # 다운로드 및 요약 통계
        download_col1, download_col2 = st.columns([1, 3])
        
        with download_col1:
            csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="데이터 다운로드",
                data=csv,
                file_name='sales_analysis_data.csv',
                mime='text/csv'
            )
        
        with download_col2:
            st.write(f"필터링된 데이터: {len(filtered_df)}건 | 표시 중: {len(display_df)}건")

# 아보카도 시장 분석
elif page == "아보카도 시장 분석":
    st.header("아보카도 시장 분석")
    
    # 데이터 로딩
    df = load_avocado_data()
    
    if df is None:
        st.info("아보카도 데이터를 로딩할 수 없습니다. 파일을 확인하고 다시 시도해주세요.")
    else:
        # 사이드바 필터
        st.sidebar.subheader("분석 필터")
        selected_type = st.sidebar.multiselect("타입 선택:", df['type'].unique(), default=df['type'].unique())
        selected_regions = st.sidebar.multiselect("지역 선택:", df['geography'].unique()[:10], default=df['geography'].unique()[:5])
        
        # 연도 범위 선택
        year_range = st.sidebar.slider("연도 범위:", 
                                      int(df['year'].min()), 
                                      int(df['year'].max()), 
                                      (int(df['year'].min()), int(df['year'].max())))
        
        # 필터링된 데이터
        filtered_df = df[
            (df['type'].isin(selected_type)) & 
            (df['geography'].isin(selected_regions)) &
            (df['year'] >= year_range[0]) & 
            (df['year'] <= year_range[1])
        ]
        
        # KPI 메트릭 (6열 레이아웃)
        with st.container():
            st.subheader("아보카도 시장 대시보드")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
            
            with kpi_col1:
                avg_price = filtered_df['average_price'].mean()
                st.metric("평균 가격", f"${avg_price:.2f}")
            
            with kpi_col2:
                total_volume = filtered_df['total_volume'].sum()
                st.metric("총 판매량", f"{total_volume:,.0f}")
            
            with kpi_col3:
                max_price = filtered_df['average_price'].max()
                st.metric("최고 가격", f"${max_price:.2f}")
            
            with kpi_col4:
                min_price = filtered_df['average_price'].min()
                st.metric("최저 가격", f"${min_price:.2f}")
            
            with kpi_col5:
                active_regions = filtered_df['geography'].nunique()
                st.metric("활성 지역", f"{active_regions}개")
            
            with kpi_col6:
                organic_ratio = (filtered_df[filtered_df['type'] == 'organic']['total_volume'].sum() / 
                               filtered_df['total_volume'].sum() * 100)
                st.metric("유기농 비율", f"{organic_ratio:.1f}%")
        
        # 분석 탭
        tab1, tab2, tab3, tab4 = st.tabs([
            "가격 트렌드", "지역별 분석", "유기농 vs 일반", "품종별 분석"
        ])
        
        with tab1:
            st.subheader("가격 트렌드 분석")
            
            # 시계열 가격 트렌드 (2열 레이아웃)
            trend_col1, trend_col2 = st.columns(2)
            
            with trend_col1:
                st.markdown("**월별 평균 가격 추이**")
                monthly_data = filtered_df.groupby([filtered_df['date'].dt.to_period('M'), 'type'])['average_price'].mean().reset_index()
                monthly_data['date'] = monthly_data['date'].astype(str)
                
                fig = px.line(monthly_data, x='date', y='average_price', color='type',
                             title="월별 평균 가격 추이")
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with trend_col2:
                st.markdown("**연도별 가격 박스플롯**")
                fig = px.box(filtered_df, x='year', y='average_price', color='type',
                           title="연도별 가격 분포")
                st.plotly_chart(fig, use_container_width=True)
            
            # 가격 분포 히스토그램 (전체 너비)
            with st.container():
                st.markdown("**가격 분포 분석**")
                price_detail_col1, price_detail_col2 = st.columns(2)
                
                with price_detail_col1:
                    fig = px.histogram(filtered_df, x='average_price', color='type',
                                     title="가격 분포", nbins=30)
                    st.plotly_chart(fig, use_container_width=True)
                
                with price_detail_col2:
                    # 가격 vs 판매량 산점도
                    fig = px.scatter(filtered_df, x='average_price', y='total_volume', 
                                   color='type', size='total_bags',
                                   title="가격 vs 판매량 관계")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("지역별 시장 분석")
            
            # 지역별 분석 (2열 레이아웃)
            region_col1, region_col2 = st.columns(2)
            
            with region_col1:
                st.markdown("**지역별 평균 가격**")
                region_price = filtered_df.groupby('geography')['average_price'].mean().sort_values(ascending=False).head(15)
                fig = px.bar(x=region_price.values, y=region_price.index, orientation='h',
                           title="상위 15개 지역 평균 가격")
                st.plotly_chart(fig, use_container_width=True)
            
            with region_col2:
                st.markdown("**지역별 총 판매량**")
                region_volume = filtered_df.groupby('geography')['total_volume'].sum().sort_values(ascending=False).head(15)
                fig = px.bar(x=region_volume.values, y=region_volume.index, orientation='h',
                           title="상위 15개 지역 총 판매량")
                st.plotly_chart(fig, use_container_width=True)
            
            # 지역별 상세 테이블
            with st.container():
                st.markdown("**지역별 종합 분석**")
                region_summary = filtered_df.groupby('geography').agg({
                    'average_price': ['mean', 'std'],
                    'total_volume': ['sum', 'mean'],
                    'total_bags': 'sum'
                }).round(2)
                
                region_summary.columns = ['평균가격', '가격표준편차', '총판매량', '평균판매량', '총포장수']
                region_summary = region_summary.sort_values('총판매량', ascending=False)
                st.dataframe(region_summary.head(20), use_container_width=True)
        
        with tab3:
            st.subheader("유기농 vs 일반 아보카도 비교")
            
            # 유기농 vs 일반 비교 (3열 레이아웃)
            organic_col1, organic_col2, organic_col3 = st.columns(3)
            
            with organic_col1:
                st.markdown("**타입별 평균 가격**")
                type_price = filtered_df.groupby('type')['average_price'].mean()
                fig = px.bar(x=type_price.index, y=type_price.values,
                           title="타입별 평균 가격 비교")
                st.plotly_chart(fig, use_container_width=True)
            
            with organic_col2:
                st.markdown("**타입별 판매량 비율**")
                type_volume = filtered_df.groupby('type')['total_volume'].sum()
                fig = px.pie(values=type_volume.values, names=type_volume.index,
                           title="타입별 판매량 비율")
                st.plotly_chart(fig, use_container_width=True)
            
            with organic_col3:
                st.markdown("**월별 타입별 가격 차이**")
                monthly_type = filtered_df.groupby([filtered_df['date'].dt.month, 'type'])['average_price'].mean().reset_index()
                monthly_type['month'] = monthly_type['date']
                
                conventional = monthly_type[monthly_type['type'] == 'conventional']['average_price'].values
                organic = monthly_type[monthly_type['type'] == 'organic']['average_price'].values
                
                if len(conventional) > 0 and len(organic) > 0:
                    price_diff = pd.DataFrame({
                        '월': range(1, min(len(conventional), len(organic)) + 1),
                        '가격차이': organic[:len(conventional)] - conventional[:len(organic)]
                    })
                    fig = px.bar(price_diff, x='월', y='가격차이',
                               title="월별 유기농-일반 가격 차이")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("품종별 상세 분석")
            
            # 품종별 분석 (2열 레이아웃)
            variety_col1, variety_col2 = st.columns(2)
            
            with variety_col1:
                st.markdown("**품종별 판매량 비교**")
                variety_data = filtered_df[['4046', '4225', '4770']].sum()
                fig = px.bar(x=variety_data.index, y=variety_data.values,
                           title="품종별 총 판매량")
                st.plotly_chart(fig, use_container_width=True)
            
            with variety_col2:
                st.markdown("**봉지 크기별 분포**")
                bag_data = filtered_df[['small_bags', 'large_bags', 'xlarge_bags']].sum()
                fig = px.pie(values=bag_data.values, names=bag_data.index,
                           title="봉지 크기별 분포")
                st.plotly_chart(fig, use_container_width=True)
            
            # 상관관계 분석
            with st.container():
                st.markdown("**변수간 상관관계 분석**")
                corr_cols = ['average_price', 'total_volume', '4046', '4225', '4770', 
                           'total_bags', 'small_bags', 'large_bags']
                corr_matrix = filtered_df[corr_cols].corr()
                
                fig = px.imshow(corr_matrix.values,
                              x=corr_matrix.columns,
                              y=corr_matrix.index,
                              title="아보카도 시장 변수간 상관관계",
                              color_continuous_scale='RdBu_r',
                              aspect='auto')
                st.plotly_chart(fig, use_container_width=True)
        
        # 원본 데이터 다운로드
        st.markdown("---")
        with st.container():
            download_col1, download_col2 = st.columns([1, 4])
            
            with download_col1:
                csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="필터링된 데이터 다운로드",
                    data=csv,
                    file_name='avocado_filtered_data.csv',
                    mime='text/csv'
                )
            
            with download_col2:
                st.write(f"필터링된 데이터: {len(filtered_df):,}건 | 전체 데이터: {len(df):,}건")

# 데이터 업로드
elif page == "데이터 업로드":
    st.header("CSV 데이터 업로드 및 분석")
    
    # 업로드 인터페이스 컨테이너
    with st.container():
        upload_col1, upload_col2, upload_col3 = st.columns([1, 2, 1])
        
        with upload_col2:
            st.markdown("""
            <div style='text-align: center; background-color: #747474; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <h4>CSV 파일 업로드</h4>
                <p>분석하고 싶은 CSV 파일을 업로드하면 자동으로 데이터를 분석해드립니다</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("CSV 파일을 선택하세요", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # 데이터 로딩
            df = pd.read_csv(uploaded_file)
            
            # 데이터 개요 (4열 레이아웃)
            with st.container():
                st.subheader("데이터 개요")
                overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
                
                with overview_col1:
                    st.metric("총 행 수", f"{len(df):,}")
                with overview_col2:
                    st.metric("총 열 수", f"{len(df.columns):,}")
                with overview_col3:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    st.metric("수치형 열", f"{len(numeric_cols)}")
                with overview_col4:
                    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                    st.metric("메모리 사용량", f"{memory_usage:.2f} MB")
            
            # 분석 탭
            tab1, tab2, tab3, tab4 = st.tabs([
                "데이터 미리보기", "시각화 분석", "데이터 품질", "고급 분석"
            ])
            
            with tab1:
                st.subheader("데이터 미리보기")
                
                # 기본 정보
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.write("**데이터 타입 정보**")
                    st.write(df.dtypes.to_frame('데이터타입'))
                
                with info_col2:
                    with st.expander("컬럼 상세 정보"):
                        st.write("**결측값 현황**")
                        missing_data = df.isnull().sum()
                        missing_df = pd.DataFrame({
                            '컬럼명': missing_data.index,
                            '결측값 수': missing_data.values,
                            '결측값 비율(%)': (missing_data.values / len(df) * 100).round(2)
                        })
                        st.dataframe(missing_df, use_container_width=True)
                
                # 데이터 미리보기
                st.write("**상위 10개 행**")
                st.dataframe(df.head(10), use_container_width=True)
                
                # 기술통계
                if len(numeric_cols) > 0:
                    st.write("**수치형 데이터 요약통계**")
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            with tab2:
                st.subheader("시각화 분석")
                
                # 시각화 옵션 선택
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                
                if len(numeric_columns) > 0:
                    vis_col1, vis_col2 = st.columns([1, 3])
                    
                    with vis_col1:
                        st.markdown("**시각화 옵션**")
                        selected_numeric = st.selectbox("수치형 컬럼:", numeric_columns)
                        
                        chart_type = st.radio("차트 타입:", [
                            "히스토그램", "박스플롯", "산점도 매트릭스"
                        ])
                        
                        if len(categorical_columns) > 0 and chart_type == "히스토그램":
                            group_by = st.selectbox("그룹화 기준:", [None] + categorical_columns)
                        else:
                            group_by = None
                    
                    with vis_col2:
                        # 차트 생성
                        if chart_type == "히스토그램":
                            if group_by:
                                fig = px.histogram(df, x=selected_numeric, color=group_by,
                                                 title=f"{selected_numeric} 분포 ({group_by}별)")
                            else:
                                fig = px.histogram(df, x=selected_numeric,
                                                 title=f"{selected_numeric} 분포")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif chart_type == "박스플롯":
                            if group_by:
                                fig = px.box(df, x=group_by, y=selected_numeric,
                                           title=f"{selected_numeric} 박스플롯 ({group_by}별)")
                            else:
                                fig = px.box(df, y=selected_numeric,
                                           title=f"{selected_numeric} 박스플롯")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif chart_type == "산점도 매트릭스":
                            if len(numeric_columns) > 1:
                                selected_cols = st.multiselect(
                                    "비교할 컬럼들:", numeric_columns, 
                                    default=numeric_columns[:min(4, len(numeric_columns))]
                                )
                                if len(selected_cols) > 1:
                                    fig = px.scatter_matrix(df[selected_cols],
                                                          title="수치형 변수들의 산점도 매트릭스")
                                    st.plotly_chart(fig, use_container_width=True)
                
                # 상관관계 분석
                if len(numeric_columns) > 1:
                    with st.container():
                        st.markdown("**상관관계 히트맵**")
                        corr_matrix = df[numeric_columns].corr()
                        fig = px.imshow(corr_matrix.values,
                                      x=corr_matrix.columns,
                                      y=corr_matrix.index,
                                      title="변수간 상관관계",
                                      color_continuous_scale='RdBu_r',
                                      aspect='auto')
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info("수치형 컬럼이 없어서 시각화를 생성할 수 없습니다.")
            
            with tab3:
                st.subheader("데이터 품질 검사")
                
                quality_col1, quality_col2 = st.columns(2)
                
                with quality_col1:
                    st.markdown("**기본 통계**")
                    if len(numeric_columns) > 0:
                        st.dataframe(df[numeric_columns].describe(), use_container_width=True)
                    else:
                        st.info("수치형 데이터가 없습니다.")
                
                with quality_col2:
                    st.markdown("**결측값 현황**")
                    missing_data = df.isnull().sum()
                    if missing_data.sum() > 0:
                        missing_df = pd.DataFrame({
                            '컬럼': missing_data.index,
                            '결측값수': missing_data.values,
                            '결측률(%)': (missing_data.values / len(df) * 100).round(2)
                        })
                        missing_df = missing_df[missing_df['결측값수'] > 0]
                        st.dataframe(missing_df, use_container_width=True)
                        
                        # 결측값 시각화
                        fig = px.bar(missing_df, x='컬럼', y='결측률(%)',
                                   title="컬럼별 결측률")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("✅ 결측값이 없습니다!")
                
                # 데이터 타입별 분포
                with st.container():
                    st.markdown("**데이터 타입 분포**")
                    type_col1, type_col2, type_col3 = st.columns(3)
                    
                    with type_col1:
                        st.write(f"**수치형**: {len(numeric_columns)}개")
                        if numeric_columns:
                            st.write("• " + "\n• ".join(numeric_columns[:5]))
                            if len(numeric_columns) > 5:
                                st.write(f"• ... 외 {len(numeric_columns)-5}개")
                    
                    with type_col2:
                        st.write(f"**범주형**: {len(categorical_columns)}개")
                        if categorical_columns:
                            st.write("• " + "\n• ".join(categorical_columns[:5]))
                            if len(categorical_columns) > 5:
                                st.write(f"• ... 외 {len(categorical_columns)-5}개")
                    
                    with type_col3:
                        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
                        st.write(f"**날짜형**: {len(datetime_cols)}개")
                        if datetime_cols:
                            st.write("• " + "\n• ".join(datetime_cols))
            
            with tab4:
                st.subheader("고급 분석")
                
                # 고급 분석 옵션
                advanced_col1, advanced_col2 = st.columns([1, 2])
                
                with advanced_col1:
                    st.markdown("**분석 옵션**")
                    
                    if len(categorical_columns) > 0:
                        analysis_type = st.selectbox("분석 유형:", [
                            "컬럼별 유니크값 분석",
                            "범주형 변수 분포",
                            "교차 분석표"
                        ])
                    else:
                        analysis_type = "컬럼별 유니크값 분석"
                
                with advanced_col2:
                    if analysis_type == "컬럼별 유니크값 분석":
                        unique_analysis = pd.DataFrame({
                            '컬럼': df.columns,
                            '유니크값수': df.nunique(),
                            '가장빈번한값': [df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A' 
                                         for col in df.columns]
                        })
                        st.dataframe(unique_analysis, use_container_width=True)
                    
                    elif analysis_type == "범주형 변수 분포" and len(categorical_columns) > 0:
                        selected_cat = st.selectbox("분석할 범주형 변수:", categorical_columns)
                        value_counts = df[selected_cat].value_counts().reset_index()
                        value_counts.columns = [selected_cat, '빈도']
                        
                        fig = px.pie(value_counts.head(10), values='빈도', names=selected_cat,
                                   title=f"{selected_cat} 분포")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif analysis_type == "교차 분석표" and len(categorical_columns) > 1:
                        cat1 = st.selectbox("첫 번째 변수:", categorical_columns)
                        cat2 = st.selectbox("두 번째 변수:", [c for c in categorical_columns if c != cat1])
                        
                        if cat1 and cat2:
                            crosstab = pd.crosstab(df[cat1], df[cat2])
                            st.dataframe(crosstab, use_container_width=True)
            
            # 데이터 다운로드 섹션
            st.markdown("---")
            st.subheader("데이터 다운로드")
            
            download_col1, download_col2, download_col3 = st.columns(3)
            
            with download_col1:
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="원본 데이터 다운로드",
                    data=csv,
                    file_name=f'{uploaded_file.name}_processed.csv',
                    mime='text/csv'
                )
            
            with download_col2:
                if len(df.select_dtypes(include=[np.number]).columns) > 0:
                    summary_stats = df.describe()
                    summary_csv = summary_stats.to_csv(encoding='utf-8-sig')
                    st.download_button(
                        label="요약통계 다운로드",
                        data=summary_csv,
                        file_name=f'{uploaded_file.name}_summary.csv',
                        mime='text/csv'
                    )
                else:
                    st.info("수치형 데이터가 없어 요약통계를 생성할 수 없습니다.")
            
            with download_col3:
                missing_report = pd.DataFrame({
                    '컬럼': df.columns,
                    '결측값_수': df.isnull().sum(),
                    '결측값_비율': (df.isnull().sum() / len(df) * 100).round(2)
                })
                missing_csv = missing_report.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="결측값 보고서",
                    data=missing_csv,
                    file_name=f'{uploaded_file.name}_missing_report.csv',
                    mime='text/csv'
                )
        
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
            st.info("파일 형식을 확인하고 다시 시도해주세요.")
    
    else:
        # 사용 가이드
        st.markdown("---")
        guide_col1, guide_col2 = st.columns([1, 1])
        
        with guide_col1:
            st.markdown("""
            #### 지원 파일 형식
            - **CSV 파일** (.csv)
            - UTF-8 또는 EUC-KR 인코딩
            - 쉼표(,)로 구분된 값
            - 첫 번째 행이 헤더인 파일 권장
            
            #### 사용 팁
            - 파일 크기는 200MB 이하 권장
            - 한글 컬럼명 지원
            - 날짜 형식: YYYY-MM-DD 또는 YYYY/MM/DD
            - 결측값은 빈 칸 또는 NULL로 표시
            """)
        
        with guide_col2:
            st.markdown("""
            #### 샘플 데이터
            다음과 같은 형식의 CSV 파일을 업로드할 수 있습니다:
            
            ```csv
            이름,나이,부서,급여,입사일
            김철수,28,개발팀,4500,2020-01-15
            이영희,32,마케팅,3800,2019-03-20
            박민수,25,디자인,3200,2021-07-10
            ```
            
            업로드 후 자동으로 데이터 타입을 인식하고
            다양한 시각화와 분석 결과를 제공합니다.
            """)
            
            st.info("👆 위의 파일 업로드 영역에 CSV 파일을 드래그하거나 클릭하여 선택하세요.")

# 푸터
st.markdown("---")
st.markdown("**데이터 분석 대시보드** | Streamlit + Pandas를 활용한 Interactive 분석 도구") 