#!/usr/bin/python

from bs4 import BeautifulSoup
import MySQLdb as mdb
import pandas as pd
import numpy as np
import settings

total_ratings = 0

def main():

    try:
        con = mdb.connect('localhost', 'arenold', settings.mysql_pass, 'arenold')
        con.set_character_set('utf8')
        print('Established db con')
        df = pd.io.sql.read_frame("SELECT * FROM RSS", con, \
                              index_col=None, coerce_float=True, params=None)
    except Exception as e:
        print('Failed to get db con')
        print(e)


    filter_thanks = df['reporter_content'].map(lambda x: 'Thanks for Registering!' not in x )
    df = df[filter_thanks]


    def countTurkRatings(row):
        global total_ratings
        count = 0
        for i in range(5):
            if row['turk_rating'+str(i)] != '':
                count += 1
                total_ratings += 1

        return count

    def averageTurkRating(row):
        ratings = []
        for i in range(5):
            if row['turk_rating'+str(i)] != '':
                ratings.append(int(row['turk_rating'+str(i)]))

        return np.mean(ratings)

    df['count_turk_ratings'] = df.apply(countTurkRatings, axis=1)
    
    print 'Total number of Turk Ratings', total_ratings
    print df['count_turk_ratings'].describe()
    print
    print df['count_turk_ratings'].value_counts()
    print

    df['avg_turk_ratings'] = df.apply(averageTurkRating, axis=1)
    print df['avg_turk_ratings'].describe()
    print
    print df['avg_turk_ratings'].value_counts()
    print

    factor = pd.cut(df['avg_turk_ratings'], [0., .25, .5, .75, 1])
    print pd.value_counts(factor)

    for idx, item in df[['link', 'avg_turk_ratings', 'count_turk_ratings']].T.iteritems():
        if item['avg_turk_ratings'] > 0.5 and item['count_turk_ratings'] >= 5:
            print item['avg_turk_ratings'], item['link']

if __name__ == '__main__':
    main()
