import requests
import json


def getTweet(query):

    headers = {
        'accept': '*/*'
        ,'authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA'
        ,'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'
        ,'x-guest-token': '1428221493305962497'
    }

    variables = {
        "screen_name": query
        ,"withSafetyModeUserFields": True
        ,"withSuperFollowsUserFields": False
    }

    params = {
        'variables': json.dumps(variables)
    }

    response = requests.get("https://twitter.com/i/api/graphql/LPilCJ5f-bs3MjJJNcuuOw/UserByScreenName", headers=headers, params=params)
    data = json.loads(response.text)
    rest_id = data['data']['user']['result']['rest_id']

    cursor = ''

    for i in range(5):

        variables = {
            "userId": rest_id,
            "count": 20,
            "withTweetQuoteCount": True,
            "includePromotedContent":True,
            "withSuperFollowsUserFields": False,
            "withUserResults":True,
            "withBirdwatchPivots":False,
            "withReactionsMetadata":False,
            "withReactionsPerspective":False,
            "withSuperFollowsTweetFields":False,
            "withVoice":True
        }

        if cursor != '':
            variables["cursor"] = cursor

        params = {
            'variables': json.dumps(variables)
        }


        response2 = requests.get("https://twitter.com/i/api/graphql/PIt4K9PnUM5DP9KW_rAr0Q/UserTweets", params=params, headers=headers)
        tweet_data = json.loads(response2.text)

        # get tweeet
        for tweet in tweet_data['data']['user']['result']['timeline']['timeline']['instructions'][0]['entries']:

            try:
                print(tweet['content']['itemContent']['tweet_results']['result']['legacy']['full_text'])
            except:
                pass

        cursor = tweet_data['data']['user']['result']['timeline']['timeline']['instructions'][0]['entries'][-1]['content']['value']


getTweet("dondaeji")