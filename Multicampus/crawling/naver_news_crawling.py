import requests
from bs4 import BeautifulSoup

url = 'https://news.naver.com/main/read.nhn?mode=LSD&mid=sec&sid1=105&oid=030&aid=0002700985'

# Response [403]은 access denied
headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36/87.0.4280.88 Safari/537.36'}
res = requests.get(url, headers=headers)

bs = BeautifulSoup(res.content, 'html.parser')

bs.select('span.lnb_date')[0].get_text()
bs.select('div.press_logo a img')[0]['title']
bs.select('h3#articleTitle')[0].get_text()
bs.select('div#articleBodyContents')[0].get_text().replace('\n', '').replace('\t', '').replace('// flash 오류를 우회하기 위한 함수 추가function _flash_removeCallback() {}', '').strip()


# 위의 코드를 함수로 만들기
def parse_news(url):
  headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36/87.0.4280.88 Safari/537.36'}
  res = requests.get(url, headers=headers)
  bs = BeautifulSoup(res.content, 'html.parser')
  date = bs.select('span.lnb_date')[0].get_text()
  media = bs.select('div.press_logo a img')[0]['title']
  title = bs.select('h3#articleTitle')[0].get_text()
  content = bs.select('div#articleBodyContents')[0].get_text().replace('\n', '').replace('\t', '').replace('// flash 오류를 우회하기 위한 함수 추가function _flash_removeCallback() {}', '').strip()

  return (title, media, date, content, url)


# 조건에 맞게 키워드로 뉴스 검색하기

keyword = '메타버스'
ds = '2021.07.17'
de = '2021.07.17'
start = 1
url_format = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query={}&sort=0&photo=0&field=0&pd=3&ds={}&de={}&cluster_rank=34&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:from{}to{},a:all&start={}'
url = url_format.format(keyword, ds, de, ds.replace('.', ''), de.replace('.', ''), start)

parse_news('https://news.naver.com/main/read.naver?mode=LSD&mid=sec&sid1=101&oid=469&aid=0000623955')

# 페이지를 순회하면서 크롤링하기

def crawl_news(keyword, ds, de):
  start = 1
  li = []
  url_format = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query={}&sort=0&photo=0&field=0&pd=3&ds={}&de={}&cluster_rank=34&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:from{}to{},a:all&start={}'

  while True:
    url = url_format.format(keyword, ds, de, ds.replace('.', ''), de.replace('.', ''), start)
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36/87.0.4280.88 Safari/537.36'}
    res = requests.get(url, headers=headers)
    bs = BeautifulSoup(res.content, 'html.parser')

    start_page = int(start/10) + 1
    curr_page = int(bs.select('div.sc_page_inner a[aria-pressed="true"]')[0].get_text())

    # start_page가 curr_page보다 작으면 반복문을 빠져나옴
    if start_page != curr_page:
      break

    for a_tag in bs.select('div.news_info div.info_group a'):
      if a_tag['href'].startswith('https://news.naver.com'):
        li.append(parse_news(a_tag['href']))

    start += 10
  import pandas as pd

  df = pd.DataFrame(li, columns=('title', 'media', 'date', 'content', 'url'))
  return df
df.to_csv('{}_{}_{}.csv'.format(keyword, ds.replace('.', ''), de.replace('.', '')))