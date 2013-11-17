
import sys
import twitter
from twitter import TwitterStream, OAuth
import json

def getTwitterIterator():

    cons_key = 'KeLYaBfPbe3JTw6uujpA'
    cons_secret = 'nU9xvD8yMt8rZo8tLR4zEon4gu3wyPJkwKNP9HEY4o'
    token = '35681519-7xlmdjQMUkKReb6Ibcbrgr1XwyZ3pPgNyBQBb5dK4'
    token_secret = 'gtKypt9sdyqJFDNIivbnJJbSPSJ5yd6jbmQ1SHuqcFNwb'
    
    auth=OAuth(token, token_secret,
                           cons_key, cons_secret)
    
    twitter_stream = TwitterStream(auth=auth) 
    return twitter_stream.statuses.sample()

def fetchTweets(iterator):
    tweets = []
    n = 0
    for t in iterator:
        if 'possibly_sensitive' in t and t['lang'] == 'en':
            tweet = { 
                         'text': t['text'], 
                         'possibly_sensitive': t['possibly_sensitive'],
                         #'user': t['user'], 
                         'source': t['source'],
                         'lang': t['lang']
                     }
            n += 1
            if n % 100 == 0:
                print n,'still going!'
            tweets.append(tweet)
    
        if sys.getsizeof(tweets) > 500000:
            break
            
    return tweets

def storeTweets(tweets):
    with open('tweets141120132100.json', 'w') as outfile:
        json.dump(tweets, outfile, indent=2)

iterator = getTwitterIterator()
storeTweets(fetchTweets(iterator))
