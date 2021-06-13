# XGBoost로 Boston Housing 데이터를 학습한다.
# XGBoost for regression
# -----------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('house_prices.csv')
df.drop(['PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Alley', 'LotFrontage'], axis=1, inplace=True)

category_nonumeric = df.select_dtypes(exclude=[np.number])
features = category_nonumeric.columns

for feature in features:
  df[feature] = df[feature].fillna('None')

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())
df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean())

from sklearn.preprocessing import LabelEncoder
for feature in features:
  le = LabelEncoder()
  le = le.fit(df[feature])
  df[feature] = le.transform(df[feature])

df_features = df.drop(['SalePrice', 'Id'], axis=1)

X_np = np.array(df_features)
y_np = np.array(df_target)

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = train_test_split(X_np, y_np, test_size = 0.2)

trainX.shape, trainY.shape

# XGBoost (regressor)로 Train 데이터 세트를 학습한다.
model = XGBRegressor(objective='reg:squarederror')  # default로 학습
model.fit(trainX, trainY)

# testX[n]에 해당하는 target (price)을 추정한다.
n = 1

df = pd.DataFrame([testX[n]])
print(df)

price = model.predict(testX[n].reshape(1,-1))
print('\n추정 price = %.2f' % (price))
print('실제 price = %.2f' % (testY[n]))
print('추정 오류 = rmse(추정 price - 실제 price) = %.2f' % np.sqrt(np.square(price - testY[n])))

# 시험 데이터 전체의 오류를 R-square로 표시한다.
print('시험 데이터 전체 오류 (R2-score) = %.4f' % model.score(testX, testY))

"""
추정 price = 259956.77
실제 price = 225000.00
추정 오류 = rmse(추정 price - 실제 price) = 34956.77
시험 데이터 전체 오류 (R2-score) = 0.9163
GBM보다 XGBoost를 사용했을 때 성능이 2%정도 증가하는 걸 확인하였다.
"""
