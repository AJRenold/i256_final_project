#!/usr/bin/python

#crawler.py

from __future__ import division

import re
import time
import json,os,sys
from cPickle import load,dump

cwd = os.getcwd()
dir = cwd+'/data/test'

jsonsamp = '/tweets121120130845.json'

fileList = [dir+jsonsamp]


def main():
    Cr = ourCrawler()
    for f in fileList:
        o = open(f)
        tweets = json.load(o)
        o.close()
        Cr.crawl(tweets)
                   
            
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
        size = 0
        i = 1
        tweetCount = 1
        t0 = time.time()
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
            size = self.sizeChecker( (noneList,moreList,otherProb,outDict) )
            t1 = time.time()
            print('Tweet Count: '+str(tweetCount) + '\tFile Size: '+ str(size*(1/1048576)) + '\tFile Num: '+str(i)+ '\tTime: ' +str(t1-t0))
            tweetCount +=1
            if size >= 104857600:
                self.outPutter( (i,{'noneList': noneList,'moreList':moreList,'otherProb':otherProb,'outDict':outDict}))
                i += 1
                noneList = []
                moreList = []
                otherProb = []
                outDict = {}    
                size = 0
        self.outPutter((i,{'noneList': noneList,'moreList':moreList,'otherProb':otherProb,'outDict':outDict}))
                        

    def outPutter(self,(i,dict)):
        for k in dict:
            outName = dir+'/out/'+k+'_'+str(i)+'.pk1'
            outFile = open(outName,'wb')
            dump(dict[k],outFile)
            outFile.close()    
        
        
    def sizeChecker(self,tuple):
        size = 0
        for it in tuple:
            size = size + sys.getsizeof(it)
        return(size)
    
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


            
