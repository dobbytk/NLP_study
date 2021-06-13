from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
df = pd.read_csv('train.csv')

"""
1. PassengerId, Ticket column 제거
2. Age의 null값을 평균값으로 채우기
3. Cabin의 null값을 N으로 채우기
4. Embarked의 null값을 N으로 채우기
"""
df.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Cabin'].fillna('N', inplace=True)
df['Embarked'].fillna('N', inplace=True)
# print('데이터 세트 Null 값 개수', df.isnull().sum().sum())

# Cabin의 경우 Null값이 많았기 때문에 N이 가장 많음. 
# C23 C25 C27같이 속성값이 제대로 정리되지 않은 경우가 많아 앞 문자만 추출
df['Cabin'] = df['Cabin'].str[:1]
# print(df['Cabin'].head(3))

# Name 칼럼에서 의미있는 데이터를 추출하기 위한 작업
df['Name'] = df.Name.str.extract('([A-Za-z]+)\.')
df['Name'] = df['Name'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr', 'Jonkhee', 'Lady', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Other')
df['Name'] = df['Name'].replace('Mlle', 'Miss')
df['Name'] = df['Name'].replace('Mme', 'Mrs')
df['Name'] = df['Name'].replace('Ms', 'Miss')

# LabelEncoder를 이용해서 맵핑
from sklearn.preprocessing import LabelEncoder

features = ['Name','Cabin', 'Sex', 'Embarked']
for feature in features:
  le = LabelEncoder()
  le = le.fit(df[feature])
  df[feature] = le.transform(df[feature])

y_df = df['Survived']
X_df = df.drop('Survived', axis=1)

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = train_test_split(X_df, y_df, test_size = 0.2)

# GBMclassifier로 Train 데이터 세트를 학습한다.
model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, max_depth=3)
model.fit(trainX, trainY)

predY = model.predict(testX)
accuracy = (testY == predY).mean()
print()
print("* 시험용 데이터로 측정한 정확도 = %.2f" % accuracy)

predY = model.predict(trainX)
accuracy = (trainY == predY).mean()
print("* 학습용 데이터로 측정한 정확도 = %.2f" % accuracy)