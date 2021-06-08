import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('diabetes.csv')

feature_list = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def hist_plot(df):
  for col in feature_list:
    df[col].plot(kind='hist', bins=20).set_title('Histogram of '+ col)
    plt.show()
hist_plot(df)

# 위 컬럼들에 대한 0값의 비율 확인
zero_count = []
zero_percent = []
for col in feature_list:
  zero_num = df[df[col]==0].shape[0]
  zero_count.append(zero_num)
  zero_percent.append(np.round(zero_num/df.shape[0]*100, 2))

zero = pd.DataFrame([zero_count, zero_percent], columns=feature_list, index=['count', 'percent'])
zero

# 0값을 우선 NaN 값으로 대체
df[feature_list] = df[feature_list].replace(0, np.nan)

# 위 5개 feature에 대해 0값을 평균값으로 대체
mean_features = df[feature_list].mean()
df[feature_list] = df[feature_list].replace(np.nan, mean_features)

# feature와 target을 나누는 작업
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']

X_np = np.array(X)
y_np = np.array(y)

# feature의 scale이 편차가 심하므로 표준화 작업 - 분류니깐 target에는 적용하지 않음
scaleX = StandardScaler()

z_X = scaleX.fit_transform(X_np)


trainX, testX, trainY, testY = train_test_split(z_X, y_np, test_size = 0.2)

model = LogisticRegression()
model.fit(trainX, trainY)

diag = model.predict(testX)

print('\n테스트셋 정확도 = %.4f' % model.score(testX, testY))
print('첫 번째 환자의 당뇨병 예측 = %.2f' % diag[0])
print('첫 번째 환자의 당뇨병 실제 = %.2f' % testY[0])
