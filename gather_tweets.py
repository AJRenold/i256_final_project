import json
import sys
import twitter
from twitter import TwitterStream, OAuth

def getTwitterIterator():
    """
        Get the Twitter Stream API iterator
    """
    # Input new keys to run
    cons_key = ''
    cons_secret = ''
    token = ''
    token_secret = ''

    auth=OAuth(token, token_secret,
                           cons_key, cons_secret)

    twitter_stream = TwitterStream(auth=auth) 
    return twitter_stream.statuses.sample()

def fetchTweets(iterator):
    """
        Use the API iterator to collect tweets with a 
        possibly_sensitve flag present, when we collect
        500000 bytes, dump the tweets to json
    """
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
    with open('tweets181120131000.json', 'w') as outfile:
        json.dump(tweets, outfile, indent=2)

if __name__ == '__main__':
    iterator = getTwitterIterator()
    storeTweets(fetchTweets(iterator))
