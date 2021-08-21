import requests
import json

def getQueryTweet(query):

    headers = {
        'accept': '*/*'
        # ,'accept-encoding': 'gzip, deflate, br'
        #,'accept-language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
        ,'authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA'
        #,'cookie': '_ga=GA1.2.443444804.1615563358; _gid=GA1.2.457209665.1629116764; G_ENABLED_IDPS=google; kdt=Lp98UdnEy0tpxDqvVwU2t9lH6l5zA9HyNnOFHZsA; dnt=1; lang=en; _sl=1; personalization_id="v1_Q3SpJaz6SfmfYvkjjRLRJA=="; guest_id=v1%3A162934951370851813; undefined; G_AUTHUSER_H=0; ct0=8b308233b2c0ca99a3efe4e98ff10acb; at_check=true; des_opt_in=Y; _gcl_au=1.1.260715623.1629359697; mbox=PC#984bf879573b4efea62ffc86ea0d61bf.32_0#1692604499|session#4b5aebb35e90409b9820fd73ffbabffc#1629361541; _twitter_sess=BAh7CSIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCI%252BXaF17AToMY3NyZl9p%250AZCIlOTZlYmE3YTJiNjMyMTY4NDNlNmMzYmMwMmQ0N2Q5YjE6B2lkIiU5Yjc4%250AMDRiMTQ0NTIxMGU5YzY2NDg4N2Q0MjgwOTQ0Mw%253D%253D--417ba2db3d87ae14aa86d5674e0c94a37217a67a; external_referer=padhuUp37zixoA2Yz6IlsoQTSjz5FgRcKMoWWYN3PEQ%3D|0|8e8t2xd8A2w%3D; gt=1428266952410034177'
        #,'referer': 'https://twitter.com/search?q=%EB%A9%94%ED%83%80%EB%B2%84%EC%8A%A4&src=typed_query&f=live'
        #,'sec-ch-ua': '"Chromium";v="92", " Not A;Brand";v="99", "Google Chrome";v="92"'
        #,'sec-ch-ua-mobile': '?0'
        #,'sec-fetch-dest': 'empty'
        #,'sec-fetch-mode': 'cors'
        #,'sec-fetch-site': 'same-origin'
        ,'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'
        #,'x-csrf-token': '8b308233b2c0ca99a3efe4e98ff10acb'
        ,'x-guest-token': '1428266952410034177'
        #,'x-twitter-active-user': 'yes'
        #,'x-twitter-client-language': 'ko'
    }

    cursor = ''

    for i in range(5):

        print("")
        print(i)
        print("")

        params = {
            'include_profile_interstitial_type': 1
            ,'include_blocking': 1
            ,'include_blocked_by': 1
            ,'include_followed_by': 1
            ,'include_want_retweets': 1
            ,'include_mute_edge': 1
            ,'include_can_dm': 1
            ,'include_can_media_tag': 1
            ,'skip_status': 1
            ,'cards_platform': 'Web-12'
            ,'include_cards': 1
            ,'include_ext_alt_text': True
            ,'include_quote_count': True
            ,'include_reply_count': 1
            ,'tweet_mode': 'extended'
            ,'include_entities': True
            ,'include_user_entities': True
            ,'include_ext_media_color': True
            ,'include_ext_media_availability': True
            ,'send_error_codes': True
            ,'simple_quoted_tweet': True
            ,'q': query
            ,'tweet_search_mode': 'live'
            ,'count': 20
            ,'query_source': 'typed_query'
            ,'pc': 1
            ,'spelling_corrections': 1
            ,'ext': 'mediaStats,highlightedLabel,voiceInfo'
        }

        if cursor != '':
            params['cursor'] = cursor

        response = requests.get("https://twitter.com/i/api/2/search/adaptive.json", headers=headers, params=params)
        data = response.json()

        for tweet in data['globalObjects']['tweets']:
            print(data['globalObjects']['tweets'][tweet]['full_text'])

        if cursor == '':
            cursor = data['timeline']['instructions'][0]['addEntries']['entries'][-1]['content']['operation']['cursor']['value']
        else:
            cursor = data['timeline']['instructions'][-1]['replaceEntry']['entry']['content']['operation']['cursor']['value']


getQueryTweet("BTS")