from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/My Drive/Colab Notebooks'

# 학습 데이터를 읽어온다.
# review -> TFIDF(문장을 수치화) -> 로지스틱 회귀(binary classification)
with open('popcorn.pkl', 'rb') as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

# 학습 - 파이프라인 사용
pipe = Pipeline([('tf_vect', TfidfVectorizer(max_df=50, max_features=10000)),
                  ('lr_clf', LogisticRegression(max_iter = 500, C = 10))])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
pred_probs = pipe.predict_proba(X_test)[:, 1]

print('정확도 =', accuracy_score(y_test, pred), ' ROC-AUC =', roc_auc_score(y_test, pred_probs))

print(len(pipe['tf_vect'].vocabulary_))

pipe

