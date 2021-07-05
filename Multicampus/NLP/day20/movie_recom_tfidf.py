import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('tmdb_5000_movies.csv')

df.head()


df['overview'][0]


# 결측값 확인 및 제거
df['overview'].isnull().sum() # 3
df['overview'] = df['overview'].fillna('')
df['overview'].isnull().sum()


# tfidf_matrix 구성
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])
print(tfidf_matrix.shape)


# avatar의 index값 
avatar_idx = df[df['title'] == 'Avatar'].index[0] # result: 0

# 코사인 유사도 구하기
movie_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(movie_sim)
print(movie_sim.shape)

# avatar의 index값으로 각 영화와의 유사도값 가져오기
avatar_sim = movie_sim[avatar_idx]
print(avatar_sim)
# 유사도가 높은 순서대로 10개의 인덱스값 가져오기
recommend = np.array(avatar_sim).argsort()[::-1][1:11]
recommend


n = 1
for idx in recommend:
  recommend_title = df['original_title'].iloc[idx]
  print('유사한 영화 {0}순위 {1}'.format(n, recommend_title))
  n += 1
def movie_recommend(cosine_matrix, movie_title):
  movie_idx = df[df['title'] == movie_title].index[0]
  movie_sim = cosine_matrix[movie_idx]
  recommend = np.array(movie_sim).argsort()[::-1][1:11]
  
  n = 1
  for idx in recommend:
    recommend_title = df['original_title'].iloc[idx]
    print('유사한 영화 {0}순위 {1}'.format(n, recommend_title))
    n += 1
  # return True

movie_recommend(movie_sim, 'Spectre')