"""
머신러닝 기초 연습 - 아보카도 가격 예측
데이터 분석 연습용 리포지토리를 위한 머신러닝 기초 학습
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
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

def load_and_prepare_data():
    """1단계: 데이터 로딩 및 전처리"""
    print("=== 1단계: 데이터 로딩 및 전처리 ===")
    
    # 데이터 로딩
    df = pd.read_csv('data/avocado.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # 특성 엔지니어링
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # 범주형 변수 인코딩
    le_type = LabelEncoder()
    le_geography = LabelEncoder()
    
    df['type_encoded'] = le_type.fit_transform(df['type'])
    df['geography_encoded'] = le_geography.fit_transform(df['geography'])
    
    print(f"데이터 크기: {df.shape}")
    print(f"특성 수: {len(df.columns)}")
    
    return df, le_type, le_geography

def feature_engineering(df):
    """2단계: 특성 엔지니어링"""
    print("\n=== 2단계: 특성 엔지니어링 ===")
    
    # 새로운 특성 생성
    df['price_volume_ratio'] = df['average_price'] / df['total_volume']
    df['total_bags_ratio'] = df['total_bags'] / df['total_volume']
    df['small_bags_ratio'] = df['small_bags'] / df['total_bags']
    df['large_bags_ratio'] = df['large_bags'] / df['total_bags']
    
    # 품종별 비율
    df['4046_ratio'] = df['4046'] / df['total_volume']
    df['4225_ratio'] = df['4225'] / df['total_volume']
    df['4770_ratio'] = df['4770'] / df['total_volume']
    
    # 결측치 처리 (0으로 나누기 방지)
    ratio_columns = ['price_volume_ratio', 'total_bags_ratio', 'small_bags_ratio', 
                    'large_bags_ratio', '4046_ratio', '4225_ratio', '4770_ratio']
    
    for col in ratio_columns:
        df[col] = df[col].fillna(0)
        df[col] = df[col].replace([np.inf, -np.inf], 0)
    
    print("생성된 새로운 특성들:")
    for col in ratio_columns:
        print(f"- {col}: {df[col].mean():.4f} (평균)")
    
    return df

def prepare_features(df):
    """3단계: 모델링을 위한 특성 준비"""
    print("\n=== 3단계: 특성 준비 ===")
    
    # 수치형 특성 선택
    feature_columns = [
        'total_volume', '4046', '4225', '4770', 'total_bags', 'small_bags', 'large_bags', 'xlarge_bags',
        'type_encoded', 'geography_encoded', 'year', 'month', 'quarter', 'day_of_week',
        'price_volume_ratio', 'total_bags_ratio', 'small_bags_ratio', 'large_bags_ratio',
        '4046_ratio', '4225_ratio', '4770_ratio'
    ]
    
    X = df[feature_columns]
    y = df['average_price']
    
    print(f"입력 특성 수: {X.shape[1]}")
    print(f"타겟 변수: average_price")
    print(f"데이터 포인트 수: {X.shape[0]}")
    
    return X, y

def train_models(X, y):
    """4단계: 모델 훈련"""
    print("\n=== 4단계: 모델 훈련 ===")
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 모델 정의
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name} 훈련 중...")
        
        # 모델 훈련
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 성능 평가
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 교차 검증
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results, X_test, y_test

def analyze_feature_importance(results, X, y):
    """5단계: 특성 중요도 분석"""
    print("\n=== 5단계: 특성 중요도 분석 ===")
    
    # Random Forest의 특성 중요도
    rf_model = results['Random Forest']['model']
    feature_importance = rf_model.feature_importances_
    feature_names = X.columns
    
    # 중요도 순으로 정렬
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("상위 10개 중요 특성:")
    print(importance_df.head(10))
    
    # 시각화
    plt.figure(figsize=(12, 8))
    
    # 특성 중요도
    plt.subplot(2, 2, 1)
    top_features = importance_df.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title('특성 중요도 (Random Forest)')
    plt.xlabel('중요도')
    
    # 모델 성능 비교
    plt.subplot(2, 2, 2)
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    
    bars = plt.bar(model_names, r2_scores)
    plt.title('모델 성능 비교 (R²)')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    
    # 색상 추가
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # RMSE 비교
    plt.subplot(2, 2, 3)
    rmse_scores = [results[name]['rmse'] for name in model_names]
    
    bars = plt.bar(model_names, rmse_scores)
    plt.title('모델 성능 비교 (RMSE)')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 예측 vs 실제 (최고 성능 모델)
    plt.subplot(2, 2, 4)
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]['model']
    
    # 테스트 데이터로 예측
    X_test_sample = X.sample(n=1000, random_state=42)
    y_pred_sample = best_model.predict(X_test_sample)
    y_true_sample = y.sample(n=1000, random_state=42)
    
    plt.scatter(y_true_sample, y_pred_sample, alpha=0.5)
    plt.plot([y_true_sample.min(), y_true_sample.max()], 
             [y_true_sample.min(), y_true_sample.max()], 'r--', lw=2)
    plt.xlabel('실제 가격')
    plt.ylabel('예측 가격')
    plt.title(f'예측 vs 실제 ({best_model_name})')
    
    plt.tight_layout()
    plt.savefig('ml_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df

def create_prediction_example(results, df, le_type, le_geography):
    """6단계: 예측 예시"""
    print("\n=== 6단계: 예측 예시 ===")
    
    # 최고 성능 모델 선택
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]['model']
    
    print(f"최고 성능 모델: {best_model_name}")
    
    # 예시 데이터 생성
    example_data = {
        'total_volume': [100000],
        '4046': [30000],
        '4225': [40000],
        '4770': [5000],
        'total_bags': [25000],
        'small_bags': [15000],
        'large_bags': [8000],
        'xlarge_bags': [2000],
        'type_encoded': [le_type.transform(['conventional'])[0]],
        'geography_encoded': [le_geography.transform(['California'])[0]],
        'year': [2020],
        'month': [6],
        'quarter': [2],
        'day_of_week': [0],
        'price_volume_ratio': [0.00001],
        'total_bags_ratio': [0.25],
        'small_bags_ratio': [0.6],
        'large_bags_ratio': [0.32],
        '4046_ratio': [0.3],
        '4225_ratio': [0.4],
        '4770_ratio': [0.05]
    }
    
    example_df = pd.DataFrame(example_data)
    
    # 예측
    predicted_price = best_model.predict(example_df)[0]
    
    print(f"\n예시 예측:")
    print(f"입력 조건: California, conventional, 2020년 6월")
    print(f"예측 가격: ${predicted_price:.2f}")
    
    # 실제 데이터와 비교
    similar_data = df[
        (df['geography'] == 'California') & 
        (df['type'] == 'conventional') & 
        (df['year'] == 2020) & 
        (df['month'] == 6)
    ]
    
    if len(similar_data) > 0:
        actual_avg = similar_data['average_price'].mean()
        print(f"실제 평균 가격: ${actual_avg:.2f}")
        print(f"예측 오차: ${abs(predicted_price - actual_avg):.2f}")

def main():
    """메인 실행 함수"""
    print("머신러닝 기초 연습 - 아보카도 가격 예측을 시작합니다!")
    print("=" * 60)
    
    # 1단계: 데이터 로딩
    df, le_type, le_geography = load_and_prepare_data()
    
    # 2단계: 특성 엔지니어링
    df = feature_engineering(df)
    
    # 3단계: 특성 준비
    X, y = prepare_features(df)
    
    # 4단계: 모델 훈련
    results, X_test, y_test = train_models(X, y)
    
    # 5단계: 특성 중요도 분석
    importance_df = analyze_feature_importance(results, X, y)
    
    # 6단계: 예측 예시
    create_prediction_example(results, df, le_type, le_geography)
    
    print("\n=== 머신러닝 연습 완료! ===")
    print("생성된 파일:")
    print("- ml_analysis.png: 머신러닝 분석 결과")

if __name__ == "__main__":
    main() 