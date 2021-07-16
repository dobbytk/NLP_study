
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pickle

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/My Drive/Colab Notebooks'

# 학습 데이터를 읽어온다.
with open('./qqp.pkl', 'rb') as f:
    q1_data, q2_data, labels, word2idx = pickle.load(f)

# question1과 question2를 하나의 쌍으로 만든다.
train_input = np.stack((q1_data, q2_data), axis=1)

# 학습 데이터와 시험 데이터로 나눈다.
trainX, testX, trainY, testY = train_test_split(train_input, labels, test_size=0.2)

train_input[0]

# XGBoost로 학습한다.
trainD = xgb.DMatrix(trainX.sum(axis=1), label = trainY)
testD = xgb.DMatrix(testX.sum(axis=1), label = testY)
data_list = [(trainD, 'train'), (testD, 'valid')]

param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'binary:logistic',
    'eval_metric': 'rmse'}

model = xgb.train(params = param, dtrain = trainD, 
                  num_boost_round = 1000,
                  evals = data_list,
                  early_stopping_rounds=10)

# 시험 데이터로 정확도를 계산한다
pred = model.predict(testD)
predY = np.where(pred > 0.5, 1, 0)
accuracy = (testY == predY).mean()
print()
print("* 시험용 데이터로 측정한 정확도 = %.2f" % accuracy)
print("* ROC AUC score = %.2f" % (roc_auc_score(testY, pred)))