import requests
import json
from bs4 import BeautifulSoup

results = []
page = 1

while page < 3:

    response = requests.get('https://www.rocketpunch.com/api/jobs/template?page=' + str(page) + '&q=')
    data = json.loads(response.text)
    bs = BeautifulSoup(data['data']['template'], "html.parser")

    for company in bs.select(".company.item"):

        result = {
            'name': company.select_one(".company-name strong").text
            ,'description': company.select_one(".description").text
            ,'jobs': []
        }

        #name = company.select_one(".company-name strong").text
        #description = company.select_one(".description").text

        for job in company.select(".job-detail > div > a.job-title"):
            result['jobs'].append(job.text)

        results.append(result)

    print(results)
    page += 1
