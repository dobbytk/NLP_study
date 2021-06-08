import pymysql
import pandas as pd

df = pd.read_excel('./seoul_uni.xlsx') # 파일 경로 다를 수 있음
conn, cur = None, None
data1, data2, data3 = "", "", ""
sql = ""

conn = pymysql.connect(host='localhost', user='root', password='1111', db = 'seoulunivdb', charset='utf8')

cur = conn.cursor()

cur.execute("create table if not exists univTable (name char(20), latitude varchar(100), longitude varchar(100))") # latitude longitude 데이터타입 변경

for i in range(len(df)):
    name = df["대학교"][i]
    latitude = df["위도"][i]
    longitude = df["경도"][i]
    cur.execute(f"insert into univTable values('{name}', '{latitude}', '{longitude}')")


conn.commit()
conn.close()