#!/usr/bin/python

#crawler.py

import re

class ourCrawler:
    from reporter import Reporter
    r = Reporter()
    def __init__(self):
        self.isCrawler = True
    def crawl(self,link,rep=r):
        rep.read(url=link)
        result = rep.report_news()
        #r = r'{.*}'
        r = r'[_{}]'
        p = re.compile(r)
        m = p.findall(result)
        if m:
            return None
        else:
            return(result)

