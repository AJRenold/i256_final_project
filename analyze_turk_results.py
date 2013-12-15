#!/usr/bin/python

from bs4 import BeautifulSoup
import MySQLdb as mdb
import numpy as np
import pandas as pd
import settings

total_ratings = 0

def main():
    """
        Output analysis of the Amazon Mechanical Turk Results
    """

    # Get database connection and load dataframe
    try:
        con = mdb.connect('localhost', 'arenold', settings.mysql_pass, 'arenold')
        con.set_character_set('utf8')
        print('Established db con')
        df = pd.io.sql.read_frame("SELECT * FROM RSS", con, \
                              index_col=None, coerce_float=True, params=None)
    except Exception as e:
        print('Failed to get db con')
        print(e)

    # Filter dataframe rows for removed Buzzfeed pages
    filter_thanks = df['reporter_content'].map(lambda x: 'Thanks for Registering!' not in x )
    df = df[filter_thanks]

    def countTurkRatings(row):
        """
           Dataframe helper function to count the number of 
           turk ratings per row
        """
        # Also keep a global count
        global total_ratings

        count = 0
        for i in range(5):
            if row['turk_rating'+str(i)] != '':
                count += 1
                total_ratings += 1

        return count

    def averageTurkRating(row):
        """
            Dataframe helper function to compute the mean
            turk rating per row
        """
        ratings = []
        for i in range(5):
            if row['turk_rating'+str(i)] != '':
                ratings.append(int(row['turk_rating'+str(i)]))

        return np.mean(ratings)

    # add the count_turk_ratings column and output descriptions
    df['count_turk_ratings'] = df.apply(countTurkRatings, axis=1)
    print 'Total number of Turk Ratings', total_ratings
    print df['count_turk_ratings'].describe()
    print
    print df['count_turk_ratings'].value_counts()
    print

    # add the avg_turk_ratings column and output descriptions
    df['avg_turk_ratings'] = df.apply(averageTurkRating, axis=1)
    print df['avg_turk_ratings'].describe()
    print
    print df['avg_turk_ratings'].value_counts()
    print

    # bin and print the counts of avg_turk_ratings column
    factor = pd.cut(df['avg_turk_ratings'], [0., .25, .5, .75, 1])
    print pd.value_counts(factor)

    # print links where the avg rating is > 0.5 and there are at least 5 turk ratings
    for idx, item in df[['link', 'avg_turk_ratings', 'count_turk_ratings']].T.iteritems():
        if item['avg_turk_ratings'] > 0.5 and item['count_turk_ratings'] >= 5:
            print item['avg_turk_ratings'], item['link']

if __name__ == '__main__':
    main()
