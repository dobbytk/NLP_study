import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# 데이터를 읽어온다.
df = pd.read_csv('train.csv')
df.head(3)

# df.info()
df.isnull().sum()   # 결측치 개수 확인

# 결측치 처리
df['Age'].fillna(df['Age'].mean(), inplace = True)  # 평균으로 대체
df['Cabin'].fillna('N', inplace = True)
df['Embarked'].fillna('N', inplace = True)
# df.isnull().sum()   # 결측치 개수 확인

# 문자열 feature 값들의 분포 확인.
df['Pclass'].value_counts()
df['Sex'].value_counts()
df['Cabin'].value_counts()
df['Embarked'].value_counts()

# 'Cabin' feature는 첫 글자만 사용하기로 한다.
df['Cabin'] = df['Cabin'].str[:1]
df['Cabin'].value_counts()

# Name feature의 호칭 (title)을 발췌한다.
name = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
name.value_counts()

title = ['Mr', 'Miss', 'Mrs', 'Master']
df['Title'] = [x if x in title else 'Other' for x in name]
df['Title'].value_counts()

# 불필요한 feature를 제거한다.
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
df.head()

# categorical feature를 one-hot encoding 한다.
df = pd.get_dummies(df, columns =['Cabin', 'Sex', 'Embarked', 'Title'])

# Z-Score 표준화
feat = ['Age', 'Fare']
df[feat] = (df[feat] - df[feat].mean()) / df[feat].std()
df.head()

# 학습 데이터를 만든다.
target_data = df['Survived']
feature_data = df.drop('Survived', axis=1)
x_train, x_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.2)

model = SVC(kernel='rbf', gamma=0.1, C=1.0, probability=True)
model.fit(x_train, y_train)

# 정확도를 측정한다
print("SVC accuracy = {0:.2f}".format(model.score(x_test, y_test)))

# prediction
y_pred = model.predict_proba(x_test)[:, 1]

# ROC curve를 그린다
fprs, tprs, thresholds = roc_curve(y_test, y_pred)

# thresholdsndarray of shape = (n_thresholds,)
# Decreasing thresholds on the decision function used to compute fpr and tpr. 
# thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1.
# lr_pred = 1.0인 경우도 있을 수 있으므로 가장 큰 threshold는 1.0보다 크게 적용한 것 같음.

plt.plot(fprs, tprs, label = 'ROC')
plt.plot([0,1], [0,1], '--', label = 'Random')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

auc = roc_auc_score(y_test, y_pred)
print("ROC AUC = {0:.4f}".format(auc))

