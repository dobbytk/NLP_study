import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 데이터를 읽어온다.
house_df = pd.read_csv('house_prices.csv')
house_df_test = pd.read_csv('house_prices_test.csv')
house_df.head()

house_df.shape
house_df.dtypes.value_counts()
isnull = house_df.isnull().sum()
isnull[isnull > 0].sort_values(ascending = False)

# 전처리
df = house_df.copy()   # 원본 데이터는 보관해 둔다.
df_test = house_df_test.copy()

# 불필요한 컬럼과 Null 값이 많은 컬럼을 삭제한다.
df.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1, inplace=True)
df_test.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1, inplace=True)

# 숫자형이 아니면 categorical로 변환하고, 숫자형은 결측치를 평균으로 대체한다.
enc = {}
for feat in df.columns:
    # categorical로 변환한다. 역변환이 가능하도록 LabelEncoder를 딕셔너리에 보관해 둔다.
    if df[feat].dtype == object:
        enc[feat] = LabelEncoder()
        df[feat] = enc[feat].fit_transform(df[feat].astype(str))
    
    # 결측치를 평균으로 대체한다.
    # elif df[feat].dtype == 'int64' or df[feat].dtype == 'float64':
    else:
        df[feat].fillna(df[feat].mean(), inplace = True)

enc_test = {}
for feat in df_test.columns:
    # categorical로 변환한다. 역변환이 가능하도록 LabelEncoder를 딕셔너리에 보관해 둔다.
    if df_test[feat].dtype == object:
        enc_test[feat] = LabelEncoder()
        df_test[feat] = enc_test[feat].fit_transform(df_test[feat].astype(str))
    
    # 결측치를 평균으로 대체한다.
    # elif df[feat].dtype == 'int64' or df[feat].dtype == 'float64':
    else:
        df_test[feat].fillna(df_test[feat].mean(), inplace = True)
print(df.head())
print(df_test.head())

# 학습 데이터와 시험 데이터를 생성한다.
y_target = df['SalePrice']
x_features = df.drop('SalePrice', axis=1)
trainX, testX, trainY, testY = train_test_split(x_features, y_target, test_size = 0.2)

# Gradient Boosting (regressor)로 Train 데이터 세트를 학습한다.
# default:
# loss = least square
# learning_rate = 0.01
# n_estimators = 500
# max_depth = 5
model = GradientBoostingRegressor(loss='ls', learning_rate=0.01, n_estimators=500, max_depth=5)
model.fit(trainX, trainY)

predictions = model.predict(df_test)
my_dict = {"Id":house_df_test['Id'], "SalePrice":predictions}
dfout = pd.DataFrame(my_dict)
dfout.to_csv('submission.csv', index=False)