import pandas as pd
from bs4 import BeautifulSoup
import requests
import ssl
import pymysql, json, calendar, time
from datetime import datetime
from threading import Timer

ssl._create_default_https_context = ssl._create_unverified_context

class DBUpdater:
    def __init__(self): # 생성자 
        """생성자: MySQL 연결 및 종목코드 딕셔너리 생성"""
        self.conn = pymysql.connect(host='localhost', user='root', password='Rnralseo15^^', db='INVESTAR', charset='utf8')

        with self.conn.cursor() as curs:
            sql = """
            create table if not exists company_info(
                code varchar(20),
                company varchar(40),
                last_update date,
                primary key (code)
            )
            """

            curs.execute(sql)
            sql = """
            create table if not exists daily_price (
                code varchar(20),
                date date,
                open bigint(20),
                high bigint(20),
                low bigint(20),
                close bigint(20),
                diff bigint(20),
                volume bigint(20),
                primary key (code, date)
            )
            """

            curs.execute(sql)
            self.conn.commit()
            
            self.codes = dict()

    def __del__(self): # 소멸자
        pass

    def read_krx_code(self):
        """KRX로부터 상장기업 목록 파일을 읽어와서 데이터프레임으로 반환 """
        url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
        krx = pd.read_html(url, header=0)[0]
        krx = krx[['종목코드', '회사명']]
        krx = krx.rename(columns={'종목코드': 'code', '회사명':'company'})
        krx.code = krx.code.map('{:06d}'.format)
        return krx

    def update_comp_info(self):
        """종목코드를 company_info 테이블에 업데이트 한 후 딕셔너리에 저장"""
        sql = "select * from company_info"
        df = pd.read_sql(sql, self.conn)
        for idx in range(len(df)): # 데이터가 len이 있다는건 한 번이라도 데이터를 DB에 밀어넣은적이 있다는 뜻.
            self.codes[df['code'].values[idx]] = df['company'].values[idx] # df['code'] -> Series, df['code'].values -> ndarray

        with self.conn.cursor() as curs:
            sql = "select max(last_update) from company_info"
            curs.execute(sql)

            rs = curs.fetchone() # fetchone()을 하면 한줄만 가져옴. fetchall()을 하면 전체를 다 리스트형태로 가져옴
            today = datetime.today().strftime('%Y-%m-%d')
            if rs[0] == None or rs[0].strftime('%Y-%m-%d') < today: # rs[0] == None -> DB 자체에 데이터가 없을 때 or 날짜가 오늘 날짜보다 이전일 때 업데이트를 실행하겠다는 뜻
                krx = self.read_krx_code()

                for idx in range(len(krx)):
                    code = krx.code.values[idx]
                    company = krx.company.values[idx]
                    # DB에 replace해서 밀어넣기
                    sql = f"replace into company_info (code, company, last"\
                        f"_update) values ('{code}', '{company}', '{today}')"
                    curs.execute(sql)

                    self.codes[code] = company
                    tmnow = datetime.now().strftime('%Y-%m-%d %H:%M')
                    # 로그 기록을 남기기 위함
                    print(f"[{tmnow}] #{idx+1:04d} replace into company_info "\
                    f"values ({code}, {company}, {today})")  
                self.conn.commit()
                print('')
        # krx = self.read_krx_code()


    def read_naver(self, code, company, pages_to_fetch):
        """네이버에서 주식 시세를 읽어서 데이터프레임으로 반환"""
        try:
            url = f"http://finance.naver.com/item/sise_day.nhn?code={code}"
            html = BeautifulSoup(requests.get(url, 
                headers={'User-agent': 'Mozilla/5.0'}).text, "lxml")
            
            pgrr = html.find("td", class_="pgRR") # td.pgRR: 맨 뒤
            if pgrr is None:
                return None
            s = str(pgrr.a["href"]).split("=") # ['/item/sise_day.nhn?code', '005930&page', '627'] 반환
            lastpage = s[-1] # 맨 마지막 페이지를 가리킴
            df = pd.DataFrame()
            # pages_to_fetch 나 크롤링 해올건데 한 페이지만 크롤링해줘 -> 1페이지에 대한 10개만 크롤링 하겠다.(페이지 단위별로 크롤링 하겠다)
            # 사용자가 입력한 크롤링하고 싶은 페이지와 실제 마지막 페이지를 비교 10페이지까지 하고 싶은데 마지막페이지가 3페이지까지 밖에 없다? 그럼 내부적으로 조정
            pages = min(int(lastpage), pages_to_fetch) 

            for page in range(1, pages + 1):
                pg_url = '{}&page={}'.format(url, page)
                df = df.append(pd.read_html(requests.get(pg_url,
                    headers={'User-agent': 'Mozilla/5.0'}).text)[0])
                tmnow = datetime.now().strftime('%Y-%m-%d %H:%M')
                print('[{}] {} ({}) : {:04d}/{:04d} pages are downloading...'.format(tmnow, company, code, page, pages), end="\r")
            
            df = df.rename(columns={'날짜':'date', '종가':'close', '전일비':'diff', '시가':'open', '고가':'high', '저가':'low', '거래량':'volume'})
            print(type(df['date']))
            df['date'].replace('.', '-')
            df = df.dropna()
            # print(df['date'])
            df[['close', 'diff', 'open', 'high', 'low', 'volume']] = df[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int) # 형 변환
            df = df[['date', 'open', 'high', 'low', 'close', 'diff', 'volume']] # 칼럼 순서 변경
        except Exception as e:
            print('Exception occured :', str(e))
            return None
        return df

    def replace_into_db(self, df, num, code, company):
        """네이버에서 읽어온 주식 시세를 DB에 REPLACE"""
        with self.conn.cursor() as curs:
            for r in df.itertuples():
                sql = f"REPLACE INTO daily_price VALUES ('{code}', "\
                    f"'{r.date}', {r.open}, {r.high}, {r.low}, {r.close}, "\
                    f"{r.diff}, {r.volume})"
                curs.execute(sql)
            self.conn.commit()
            print('[{}] #{:04d} {} ({}) : {} rows > REPLACE INTO daily_'\
                'price [OK]'.format(datetime.now().strftime('%Y-%m-%d'\
                ' %H:%M'), num+1, company, code, len(df)))
        

    def update_daily_price(self, pages_to_fetch):
        for idx, code in enumerate(self.codes):
            df = self.read_naver(code, self.codes[code], pages_to_fetch)
            if df is None:
                continue
            self.replace_into_db(df, idx, code, self.codes[code])

    def execute_daily(self):
        """실행 즉시 및 매일 오후 다섯시에 daily_price 테이블 업데이트"""
        self.update_comp_info()
        
        try:
            with open('config.json', 'r') as in_file:
                config = json.load(in_file)
                pages_to_fetch = config['pages_to_fetch']
        except FileNotFoundError:
            with open('config.json', 'w') as out_file:
                pages_to_fetch = 30 
                config = {'pages_to_fetch': 1}
                json.dump(config, out_file)
        self.update_daily_price(pages_to_fetch)

        tmnow = datetime.now()
        lastday = calendar.monthrange(tmnow.year, tmnow.month)[1]
        if tmnow.month == 12 and tmnow.day == lastday:  # 타이머를 지정해서 만약 12월 31일이면 년, 월, 일, 시간 세팅
            tmnext = tmnow.replace(year=tmnow.year+1, month=1, day=1,
                hour=17, minute=0, second=0)
        elif tmnow.day == lastday:  # 달의 마지막 날이다 월, 일, 시간만 세팅
            tmnext = tmnow.replace(month=tmnow.month+1, day=1, hour=17,
                minute=0, second=0)
        else:
            tmnext = tmnow.replace(day=tmnow.day+1, hour=17, minute=0, # 하루 지나서 그 시간에 세팅
                second=0)   
        tmdiff = tmnext - tmnow
        secs = tmdiff.seconds
        t = Timer(secs, self.execute_daily) # 타이머를 지정해서 만약 12월 31일이면 1월로 강제 세팅
        print("Waiting for next update ({}) ... ".format(tmnext.strftime
            ('%Y-%m-%d %H:%M')))
        t.start()

if __name__ == '__main__':
    dbu = DBUpdater()
    dbu.execute_daily()



# update_daily_price, update_comp_info 두 함수만 잘 기억할 것! 

"""
execute_daily -> update_comp_info   -> read_krx_code

              -> update_daily_price -> read_naver, replace_into_db

"""