#!/usr/bin/env python
# coding: utf-8

import requests
import pandas as pd
import os
from bs4 import BeautifulSoup
url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'

result = []
db = dict()
res = requests.get(url, headers={'User-agent': 'Mozilla/5.0'})
soup = BeautifulSoup(res.text, 'html.parser')

texts = soup.find_all("td")
for text in texts:
    result.append(text.text.strip())

for i in range(1, len(result), 9): # db라는 딕셔너리에 {종목:종목코드} 형태로 저장
    db[result[i-1]] = result[i]

# 네이버 금융 일별시세 크롤링
table_to_list = []
for key in db:
    for i in range(3):
        url = "https://finance.naver.com/item/sise_day.nhn?code={}&page={}".format(db[key], i+1)
        res = requests.get(url, headers={'User-agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.text, 'html.parser')

        table = soup.find("table", attrs={"class":"type2"})
        t = table.text.split() 

        header = t[:7] # 헤더만 따로 저장 - 첫 번째 칼럼에 '종목', 두 번째 칼럼에 '종목코드' 추가
        header.insert(0, '종목') 
        header.insert(1, '종목코드')
        for i in range(14, len(t)+1, 7): # 각 row를 따로 리스트에 저장 table_to_list는 2차원 리스트
            tmp = t[i-7:i] 
            tmp.insert(0, key) # 종목
            tmp.insert(1, db[key]) # 종목코드
            table_to_list.append(tmp)

#
# csv파일로 저장하는데 output.csv파일이 있으면 누적해서 저장 
df = pd.DataFrame(table_to_list)
if not os.path.exists('output.csv'):
    df.to_csv('output.csv', index=False, mode='w', header=header)
else:
    df.to_csv('output.csv', index=False, mode='a', header=False)
