#!/usr/bin/python

#crawler.py

from __future__ import division

from copy import copy
from cPickle import load,dump
from itertools import islice
import json
import os
from progressbar import ProgressBar, Percentage, Bar, ETA
import re
from reporter import Reporter
import sys
import time

def main():
    Cr = ourCrawler()

    ## Loop through files in our data_dir, crawling each data file
    data_dir = 'data/test/'
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json'):
                o = open(data_dir + f)
                tweets = json.load(o)
                o.close()
                Cr.crawl(tweets, f)

class ourCrawler:
    r = Reporter()

    def __init__(self):
        self.isCrawler = True

    def crawl(self, tweets, filename):

        file_count = 1
        t0 = time.time()
        crawl_count = 0
        crawled_tweets = []

        print 'crawling', filename
        pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(tweets)).start()

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
                    continue

                ## Save crawl information
                t['page_text'] = page_text
                t['link_followed'] = link
                t['link_actual'] = url_visited
                t['errors'] = ''

                crawled_tweets.append(t)
                crawl_count += 1
                pbar.update(crawl_count)

                size = self.sizeChecker(crawled_tweets)

                ## If file is over 10Mb output a chunk of the results
                ## 100 Mb = 104857600
                if size > 10485760:
                    t1 = time.time()
                    print('Tweet Count: '+str(len(crawled_tweets)) \
                      + '\tFile Size: '+ str(size*(1/1048576)) \
                      + '\tTime: ' +str(t1-t0))
                    print 'saving output, reseting crawled_tweets'
                    crawled_tweets = []
                    self.saveCrawl(crawled_tweets, filename, file_count)
                    file_count += 1

        if len(crawled_tweets) > 0:
            pbar.finish()
            print 'finished ' + filename + 'saving last outfile' + str(file_count)
            self.saveCrawl(crawled_tweets, filename, i)

    def saveCrawl(self, tweets, filename, file_no):
        """
        Output crawled tweets to json
        """
        with open('crawled_'+filename+str(file_no), 'w') as outfile:
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
            yield link

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
