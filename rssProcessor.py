
from bs4 import BeautifulSoup
import csv
import datetime
import os
import requests
from reporter import Reporter
from settings import mysql_pass, parser_token
import time

## We should switch from a file to a list of RSS feeds
xmlFile ='nsfw.xml'

def main():
    rsp = rssProcessor()
    outDict = rsp.process(xmlFile)
    rsp.writeFile(outDict,xmlFile)
    outDict = rsp.getDatabaseStatus(outDict)
    outDict = rsp.addCrawlData(outDict)

    print outDict.values()[0]

class rssProcessor:

    def __init__(self):
        self.isxmlProcessor = True
        self.parser_token = parser_token
        self.parser_api = 'http://www.readability.com/api/content/v1/parser?url={link}&token={token}'

    def process(self,xmlFile):
        cwd = os.getcwd()
        xmlLoc = cwd+'/data/test/buzzLinkRSS/'+xmlFile
        soup = BeautifulSoup(open(xmlLoc),["lxml", "xml"])
        outDict = {}
        items = soup.findAll('item')
        for i in items:
            link = i.link.string
            flag = 0
            medList = i.find_all('rating')
            for m in medList:
                if m.string == 'adult':
                    flag = 1
            outDict[hash(link)] = { 'link': link, 'flag': flag, 'dbstatus': False }
        return outDict

    def getDatabaseStatus(self, outDict):
        for key, value in outDict.iteritems():
            print key

        return outDict

    def addCrawlData(self, outDict):
        reporter = Reporter()
        for key, value in outDict.iteritems():
            if value['dbstatus'] == False:
                api_params = { 'link': value['link'],
                    'token': self.parser_token
                    }

                resp = requests.get(self.parser_api.format(**api_params))
                value.update(resp.json())

                if 'content' in value:
                    reporter.read(html=value['content'])
                    _, value['reporter_content'] = reporter.report_news()

                time.sleep(1)

        return outDict

    def updateDatabase(self, outDict):
        pass

    def writeFile(self, outDict, xmlFile):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%m-%d-%Y_%H_%M')
        name = xmlFile.split('.')[0]
        cwd = os.getcwd()
        with open(cwd+"/data/test/buzzLinkTest/"+name+'_'+st+'.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(('link','flag'))
            for k in outDict:
                writer.writerow((k,outDict[k]))
        csvfile.close()


main()



if __name__ == '__main__':
    main()
