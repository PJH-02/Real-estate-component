#테스트 데이터 분리
#csv 파일 읽어오기
import pandas as pd
data = pd.read_csv('강남구 최종 데이터.csv', encoding = 'cp949')
df = data.loc[:, ['지하철 거리', '학교 거리', '병원 거리', '면적당금액']]
df.columns = ['sub','school','doc','price']

#데이터 세트 분리
from sklearn.model_selection import train_test_split
x=df[['sub', 'school', 'doc']]
y=df[['price']]

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

#표준화
from sklearn.preprocessing import StandardScaler

#독립변수(X) 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 학습 데이터의 평균과 표준편차를 사용

#OLS
import statsmodels.api as sm

#statsmodels OLS 모델 학습을 위해 상수항(Intercept) 추가
X_train_scaled = sm.add_constant(X_train_scaled)  # 절편 추가
X_test_scaled = sm.add_constant(X_test_scaled)

#OLS 모델 학습
model = sm.OLS(y_train, X_train_scaled)  # OLS 모델 생성
results = model.fit()  # 모델 학습

#모델 요약 출력
print(results.summary())

# 6. 테스트 데이터 예측
y_pred = results.predict(X_test_scaled)

