#!/usr/bin/python

#crawler.py

from __future__ import division

from copy import copy
from cPickle import load,dump
from itertools import islice
import json
import logging
import MySQLdb as mdb
import os
from progressbar import ProgressBar, Percentage, Bar, ETA
import re
from reporter import Reporter
from settings import mysql_pass
import sys
import time

logging.basicConfig(filename='log_file.log',level=logging.WARNING, format='%(asctime)s %(message)s')

def main():
    r = Reporter()

    try:
        con = mdb.connect('localhost', 'arenold', mysql_pass, 'arenold')
        con.set_character_set('utf8')
        logging.warning('Established db con')
    except Exception as e:
        logging.warning('Failed to get db con')
        logging.exception(e)

    count = 1

    logging.warning('Crawling . . .')
    while count > 0:
        with con:
            cur = con.cursor(mdb.cursors.DictCursor)
            cur.execute('SELECT count(id) as count from tweets where crawled=0;')
            count = cur.fetchall()

        with con:
            cur = con.cursor(mdb.cursors.DictCursor)
            cur.execute('SET NAMES utf8;')
            cur.execute('SET CHARACTER SET utf8;')
            cur.execute('SET character_set_connection=utf8;')

            cur.execute('SELECT id, link FROM tweets where crawled=0;')
            for result in islice(cur.fetchall(),100):
                last = result['id']
                try:
                    url_visited, page_text = fetchText(result['link'], r)
                except Exception as e:
                    logging.warning(str(result['id']) +' visiting '+result['link'])
                    logging.exception(e)
                    result['errors'] = 'error visiting'
                    result['page_text'] = 'None'
                    result['link_actual'] = 'None'

                if 'errors' not in result:
                    if page_text:
                        result['page_text'] = unicode(con.escape_string(page_text.encode('utf8')), encoding='utf-8')
                    else:
                        result['page_text'] = 'None'
                    result['link_actual'] = url_visited
                    result['errors'] = 'None'

                update = u"UPDATE tweets SET page_text='{page_text}', errors='{errors}',\
    link_actual='{link_actual}', crawled=True WHERE id={id};"

                cmd = update.format(**result)
                try:
                    cur.execute(cmd)
                except Exception as e:
                    logging.warning(str(result['id']) + ' failed mysql update')
                    logging.exception(e)

            con.commit()

        logging.warning('end of last 100: '+str(last))

def crawl():
    ## loop through extractLinks generator output
    for link in self.extractLinks(t['text']):

        try: ## try to follow the link
            url_visited, page_text = self.fetchText(link)
        except Exception as e: ## if we get an error, save the link
            t['page_text'] = ''
            t['link_followed'] = link
            t['errors'] = e
            l = str(e) + ' visiting ' + link
            logging.exception(e)
            logging.warning(l)
            continue

        ## Save crawl information
        t['page_text'] = page_text
        t['link_followed'] = link
        t['link_actual'] = url_visited
        t['errors'] = ''

def fetchText(link, reporter):
    reporter.read(url=link)
    url, text = reporter.report_news()
    #r = r'{.*}'

    ## Not sure if this regex is doing what we expect
    r = r'[_{}]'
    p = re.compile(r)
    m = p.findall(text)
    if m:
        return url, None
    else:
        return url, text

class ourCrawler:
    r = Reporter()

    def __init__(self):
        self.isCrawler = True

    def crawl(self, tweets, filename):

        file_count = 1
        t0 = time.time()
        crawl_count = 0
        crawled_tweets = []

        l = 'crawling', filename
        logging.warning(l)
        #pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(tweets)).start()

        for t in islice(tweets,None): ## Use islice for testing
            ## replace None with 5 to test

            ## loop through extractLinks generator output
            for link in self.extractLinks(t['text']):

                try: ## try to follow the link
                    url_visited, page_text = self.fetchText(link)
                except Exception as e: ## if we get an error, save the link
                    t['page_text'] = ''
                    t['link_followed'] = link
                    t['errors'] = e
                    l = str(e) + ' visiting ' + link
                    logging.exception(e)
                    logging.warning(l)
                    continue

                ## Save crawl information
                t['page_text'] = page_text
                t['link_followed'] = link
                t['link_actual'] = url_visited
                t['errors'] = ''

                crawled_tweets.append(t)
                crawl_count += 1
                #pbar.update(crawl_count)

                size = self.sizeChecker(crawled_tweets)

                ## logging
                if crawl_count % 100 == 0:
                    t1 = time.time()
                    l = 'Tweet Count: '+str(len(crawled_tweets)) \
                      + '\tFile Size: '+ str(size*(1/1048576)) \
                      + '\tTime: ' +str(t1-t0)
                    logging.warning(l)

                ## If file is over 10Mb output a chunk of the results
                ## 100 Mb = 104857600
                if size > 10485760:
                    t1 = time.time()
                    l = 'Tweet Count: '+str(len(crawled_tweets)) \
                      + '\tFile Size: '+ str(size*(1/1048576)) \
                      + '\tTime: ' +str(t1-t0)
                    logging.warning(l)
                    l = 'saving output, reseting crawled_tweets'
                    logging.warning(l)
                    crawled_tweets = []
                    self.saveCrawl(crawled_tweets, filename, file_count)
                    file_count += 1

        if len(crawled_tweets) > 0:
            #pbar.finish()
            l = 'finished ' + filename + 'saving last outfile ' + str(file_count)
            logging.warning(l)
            self.saveCrawl(crawled_tweets, filename, file_count)

    def saveCrawl(self, tweets, filename, file_no):
        """
        Output crawled tweets to json
        """
        with open('crawled_'+str(file_no)+'_'+filename, 'w') as outfile:
            json.dump(tweets, outfile, indent=2)

    def sizeChecker(self,tuple):
        size = 0
        for it in tuple:
            size = size + sys.getsizeof(it)
        return(size)

    def fetchText(self,link):
        self.r.read(url=link)
        url, text = self.r.report_news()
        #r = r'{.*}'

        ## Not sure if this regex is doing what we expect
        r = r'[_{}]'
        p = re.compile(r)
        m = p.findall(text)
        if m:
            return url, None
        else:
            return url, text

    def extractLinks(self,text):

        link_pattern = re.compile(r'(https?://[\w./]*\w{1})')
        for link in link_pattern.findall(text):
            if '//' in link:
                if len(link[link.find('//')+2:]) > 5:
                    yield link
            else:
                l = 'not crawling ' + link
                logging.warning(l)

        """
        linkPat = '''http://'''
        lp = re.compile(linkPat)
        instances = lp.findall(text)
        print 'links', instances
        if len(instances) > 1:
            print 'ALERT: MORE THAN ONE LINK FOUND'
            return( (text,'more') )
        if len(instances) == 0:
            print "ALERT: DID NOT FIND ANY LINKS"
            return( (text,'none') )
        else:
            st = lp.search(text).start()
            findSpace = re.compile(''' ''')
            yesSpace = findSpace.search(text[st:]+'')
            if yesSpace:
                link = text[st:yesSpace.start()+st]
                return ( (link, 'OK') )
            else:
                link = text[st:]
                return ( (link, 'OK') )
        """

if __name__ == '__main__':
    main()
