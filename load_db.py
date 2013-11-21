#!/usr/bin/python

#crawler.py

from __future__ import division

from copy import copy
from itertools import islice
import json
import logging
import os
import MySQLdb as mdb
import re
from settings import mysql_pass

logging.basicConfig(filename='log_file.log',level=logging.WARNING, format='%(asctime)s %(message)s')

def main():

    try:
        con = mdb.connect('localhost', 'arenold', mysql_pass, 'arenold')
        con.set_character_set('utf8')
        logging.warning('Established db con')
    except Exception as e:
        logging.warning('Failed to get db con')
        logging.exception(e)

    ## Loop through files in our data_dir, crawling each data file
    data_dir = 'data/load_db/'
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json'):
                o = open(data_dir + f)
                tweets = json.load(o)
                o.close()

    with con:
        cur = con.cursor(mdb.cursors.DictCursor)
        cur.execute('SET NAMES utf8;')
        cur.execute('SET CHARACTER SET utf8;')
        cur.execute('SET character_set_connection=utf8;')

        insert = u"INSERT INTO tweets(lang, text, source, link, possibly_sensitive) \
values('{lang}', '{text}', '{source}', '{link}', '{possibly_sensitive}');"

        for t in islice(tweets,None): ## Use islice for testing
            ## replace None with 5 to test
            ## loop through extractLinks generator output
            for link in extractLinks(t['text']):
                tweet = copy(t)
                tweet['link'] = link
                tweet['text'] = unicode(con.escape_string(tweet['text'].encode('utf8')), encoding='utf-8')
                cmd = insert.format(**tweet)
                try:
                    cur.execute(cmd)
                except Exception as e:
                    logging.warning(cmd)
                    logging.exception(e)


        con.commit()

def extractLinks(text):

    link_pattern = re.compile(r'(https?://[\w./]*\w{1})')
    for link in link_pattern.findall(text):
        if '//' in link:
            if len(link[link.find('//')+2:]) > 5:
                yield link
        else:
            l = 'not crawling ' + link
            logging.warning(l)

if __name__ == '__main__':
    main()
