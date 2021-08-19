import requests
import json
import time

magazineOffset = 0
contestOffset = 0
exhibitOffset = 0
galleryOffset = 0

for i in range(10): # 0~9

    params = {
        'magazineOffset': magazineOffset
        ,'contestOffset': contestOffset
        ,'exhibitOffset': exhibitOffset
        ,'galleryOffset': galleryOffset
    }

    response = requests.get("https://www.jungle.co.kr/recent.json", params=params)
    data = json.loads(response.text)

    for d in data['moreList']:
        print(d['title'])
        print(d['targetCode'])

    magazineOffset = data['magazineOffset'] #0
    contestOffset = data['contestOffset'] #6
    exhibitOffset = data['exhibitOffset'] #0
    galleryOffset = data['galleryOffset'] #0

    time.sleep(1)

    

#print(data['moreList'])