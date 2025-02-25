from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.read_csv('강남구 최종 데이터.csv', encoding = 'cp949')
df = data.loc[:, ['지하철 거리', '학교 거리', '병원 거리', '면적당금액']]
df.columns = ['sub','school','doc','price']

X = df[['sub', 'school', 'doc']]  # 독립변수
y = df['price']  # 종속변수

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#다항식 변환 적용
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)  # train 데이터 기준으로 변환

#표준화
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)  # train 데이터 기준으로 변환

#모델 학습
model = LinearRegression()
model.fit(X_train_poly_scaled, y_train)

# 모델 평가
r2_train = model.score(X_train_poly_scaled, y_train)
r2_test = model.score(X_test_poly_scaled, y_test)
print(f"Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")

#모델 총평가
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_polynomial_regression(model, poly, scaler, X_train, X_test, y_train, y_test, feature_names):
    # 1. 특성 중요도 분석
    feature_names_poly = poly.get_feature_names_out(feature_names)
    coefficients = pd.DataFrame({
        'Feature': feature_names_poly,
        'Coefficient': model.coef_
    })
    coefficients['Abs_Coefficient'] = abs(coefficients['Coefficient'])
    coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
    
    # 2. 예측 성능 지표 계산
    y_train_pred = model.predict(X_train_poly_scaled)
    y_test_pred = model.predict(X_test_poly_scaled)
    
    metrics = {
        'Train_MSE': mean_squared_error(y_train, y_train_pred),
        'Test_MSE': mean_squared_error(y_test, y_test_pred),
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Train_MAE': mean_absolute_error(y_train, y_train_pred),
        'Test_MAE': mean_absolute_error(y_test, y_test_pred),
        'Train_R2': r2_train,
        'Test_R2': r2_test
    }
    
    # 3. 잔차 분석
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred
    
    return coefficients, metrics, residuals_train, residuals_test

# 분석 실행
feature_names = ['sub', 'school', 'doc']
coefficients, metrics, residuals_train, residuals_test = analyze_polynomial_regression(
    model, poly, scaler, X_train_poly_scaled, X_test_poly_scaled, 
    y_train, y_test, feature_names
)

# 결과 출력
print("\n=== 특성 중요도 (상위 10개) ===")
print(coefficients.head(10))

print("\n=== 모델 성능 지표 ===")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# 시각화
plt.figure(figsize=(15, 5))

# 특성 중요도 시각화
plt.subplot(121)
top_10_features = coefficients.head(10)
plt.barh(range(len(top_10_features)), top_10_features['Abs_Coefficient'])
plt.yticks(range(len(top_10_features)), top_10_features['Feature'])
plt.title('Top 10 Feature Importance')

# 잔차 분포 시각화
plt.subplot(122)
plt.hist(residuals_test, bins=30, alpha=0.5, label='Test')
plt.hist(residuals_train, bins=30, alpha=0.5, label='Train')
plt.title('Residuals Distribution')
plt.legend()

plt.tight_layout()
plt.show() 