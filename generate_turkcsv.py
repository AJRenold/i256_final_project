#!/usr/bin/python

from bs4 import BeautifulSoup
import csv
import MySQLdb as mdb
import pandas as pd
import settings
import re
from random import choice, random, shuffle
from itertools import islice

def main():

    with open('data/turk_data/turk_data_ids_1.csv', 'rb') as infile:
        turked_ids = []
        for row in csv.reader(infile):
             turked_ids.extend(row)

    # IDs previously submitted to mturk
    turked_ids = set(turked_ids)

    try:
        con = mdb.connect('localhost', 'arenold', settings.mysql_pass, 'arenold')
        con.set_character_set('utf8')
        print('Established db con')
        df = pd.io.sql.read_frame("SELECT * FROM RSS", con, \
                              index_col=None, coerce_float=True, params=None)
    except Exception as e:
        print('Failed to get db con')
        print(e)


    def getText(parser_content):
        soup = BeautifulSoup(parser_content)
        text = re.sub('\n',' ',soup.getText().strip())
        return text

    def getTextLen(content):
        return len(content.split(' '))

    print df.columns
    validate(df)

    df['content'] = df['parser_content'].apply(getText)
    df['content_len'] = df['content'].apply(getTextLen)

    print df['content_len'].describe()

    sensitive_df = df[df['sensitive_flag'] == 1]
    filter_thanks = sensitive_df['reporter_content'].map(lambda x: 'Thanks for Registering!' not in x 
                                                                    and x != 'None' )
    sensitive_df = sensitive_df[filter_thanks]

    not_sensitive_df = df[df['sensitive_flag'] == 0]
    filter_none = not_sensitive_df['reporter_content'].map(lambda x: x != 'None' )
    not_sensitive_df = not_sensitive_df[filter_none]

    print 'sensitive rows',len(sensitive_df)
    print 'not sensitive rows',len(not_sensitive_df)

    sensitive_links = sensitive_df['link'].tolist()

    ids = []
    data = []

    row = []
    row_ids = []
    for idx, item in islice(not_sensitive_df[['link','id','content_len']].T.iteritems(),None):
        link = item['link'] + '#' + str(item['id'])
        if item['content_len'] >= 100 and str(item['id']) not in turked_ids:

            if len(row) < 3:
                row.append(link)

            else:
                shuffle(row)
                data.append(row)
                row = [ link ]

    print 'data',len(data)
    print 'ids',len(ids)

    """
    with open('data/turk_data/turk_data_2.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['site1', 'site2', 'site3'])
        for row in data:
            csvwriter.writerow(row)
    """

def validate(df):
    series = df['reporter_content']
    for i in range(len(series)):
        ent = series[i]
        # print ent
        if isinstance(ent, float) == True:
            print 'float', df['id'][i]
        elif ent == 'None':
            print 'None', df['id'][i]
        elif 'Thanks for Registering!' in ent:
            print 'Thanks', df['id'][i]
        else:
            pass

if __name__ == '__main__':
    main()
