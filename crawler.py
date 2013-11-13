#!/usr/bin/python

#crawler.py

import re
import time
import json,os
from cPickle import load,dump


cwd = os.getcwd()
dir = cwd+'/data/test/'

jsonsamp = 'tweets121120130845.json'

fileList = [dir+jsonsamp]


def main():
    Cr = ourCrawler()
    for f in fileList:
        o = open(f)
        tweets = json.load(o)
        o.close()
        testSet = tweets[1:20]                  #This controls how big the output will be
        testRes = Cr.crawl(testSet)
        for k in testRes:
            outName = dir+'/out/'+k+'.pk1'
            outFile = open(outName,'wb')
            dump(testRes[k],outFile)
            outFile.close()           
            
            
            
class ourCrawler:
    from reporter import Reporter
    r = Reporter()
    
    def __init__(self):
        self.isCrawler = True
       
    def crawl(self,tweets):
        noneList = []
        moreList = []
        otherProb = []
        outDict = {}    
        for t in tweets:
            linktry = self.extractLink(t)
            if linktry[1] == 'more':
                moreList.append(t)
            if linktry[1] == 'none':
                noneList.append(t)
            if linktry[1] == 'OK':
                link = linktry[0]
                sensFlag = t['possibly_sensitive']
                time.sleep(.25)
                try:
                    linkText = self.fetchText(link)
                    outDict[link] = [linkText,sensFlag]
                except AttributeError:
                    otherProb.append(t)
        return({'noneList': noneList,'moreList':moreList,'otherProb':otherProb,'outDict':outDict})
            
    def fetchText(self,link,rep=r):
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
            
    def extractLink(self,tweet):
        text = tweet['text']
        linkPat = '''http://'''
        lp = re.compile(linkPat)
        instances = lp.findall(text)
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


        
main()            

            
