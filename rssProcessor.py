
from bs4 import BeautifulSoup
import csv
import datetime
import logging
import MySQLdb as mdb
import os
import requests
from reporter import Reporter
from settings import mysql_pass, parser_token
import time
from urllib2 import Request, urlopen, URLError, HTTPError

logging.basicConfig(filename='log_file21112013.log',level=logging.WARNING, format='%(asctime)s %(message)s')

## We should switch from a file to a list of RSS feeds. DONE.
xmlFeeds = ["index","community/justlaunched","animals","celeb","entertainment","food","tech",\
            "lgbt","music","politics","rewind","sports","lol","win","omg","cute","geeky","trashy",\
            "fail","wtf","badge/gold-star","badge/collection","badge/time-waster","ew","lists","nsfw",\
            "category/culture","category/movie","category/music","category/tv","category/celebrity",
            "category/style","category/food","category/business","category/science"]

def main():

    try:
        con = mdb.connect('localhost', 'arenold', mysql_pass, 'arenold')
        con.set_character_set('utf8')
        logging.warning('Established db con')
    except Exception as e:
        logging.warning('Failed to get db con')
        logging.exception(e)

    rsp = rssProcessor()
    for f in xmlFeeds:
        outDict = rsp.process(f)
        outDict = rsp.getDatabaseStatus(outDict, con)
        outDict = rsp.addCrawlData(outDict)
        rsp.updateDatabase(outDict, con)

class rssProcessor:

    def __init__(self):
        self.isxmlProcessor = True
        self.parser_token = parser_token
        self.parser_api = 'http://www.readability.com/api/content/v1/parser?url={link}&token={token}'

    def process(self,xmlFeed):
        xmlCont = self.fetchFeed(xmlFeed)
        soup = BeautifulSoup(xmlCont,["lxml", "xml"])
        outDict = {}
        items = soup.findAll('item')
        for i in items:
            link = i.link.string
            flag = 0
            medList = i.find_all('rating')
            for m in medList:
                if m.string == 'adult':
                    flag = 1
            outDict[str(hash(link))] = { 'link': link, 'flag': flag,
                                        'inDB': False, 'id': str(hash(link)) }
        return outDict
    
    def fetchFeed(self,xmlFeed):
        url = 'http://www.buzzfeed.com/'+xmlFeed+'.xml'
        req = Request(url)
        try:
            response = urlopen(req)
            return(response.read())
        except HTTPError as e:
            print 'The server couldn\'t fulfill the request.'
            print 'Error code: ', e.code
        except URLError as e:
            print 'We failed to reach a server.'
            print 'Reason: ', e.reason
        else:
            pass
            # everything is fine

    def getDatabaseStatus(self, outDict, con):
        with con:
            cur = con.cursor(mdb.cursors.DictCursor)
            for key, value in outDict.iteritems():
                cur.execute('SELECT count(id) as count FROM RSS WHERE id={};'.format(key))
                count = cur.fetchall()[0]['count']
                if count == 1:
                    logging.warning('in db ' + key)
                    value['inDB'] = True

            con.commit()

        return outDict

    def addCrawlData(self, outDict):
        reporter = Reporter()
        for key, value in outDict.iteritems():
            if value['inDB'] == False:
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

    def updateDatabase(self, outDict, con):
        insert = u"INSERT INTO RSS(id, link, title, parser_content, reporter_content, sensitive_flag) \
                values('{id}', '{link}', '{title}', '{content}', '{reporter_content}', '{flag}');"

        with con:
            cur = con.cursor(mdb.cursors.DictCursor)
            cur.execute('SET NAMES utf8;')
            cur.execute('SET CHARACTER SET utf8;')
            cur.execute('SET character_set_connection=utf8;')

            for value in outDict.values():
                if value['inDB'] == False:

                    value['link'] = unicode(con.escape_string(value['link']
                            .encode('utf8')), encoding='utf-8')

                    if 'title' not in value:
                        value['title'] = u'None'
                    else:
                        value['title'] = unicode(con.escape_string(value['title']
                            .encode('utf8')), encoding='utf-8')

                    if 'content' not in value:
                        value['content'] = u'None'
                        value['reporter_content'] = u'None'
                    else:
                        value['content'] = unicode(con.escape_string(value['content']
                            .encode('utf8')), encoding='utf-8')

                        value['reporter_content'] = unicode(con.escape_string(value['reporter_content']
                            .encode('utf8')), encoding='utf-8')

                    cmd = insert.format(**value)
                    try:
                        cur.execute(cmd)
                    except Exception as e:
                        logging.warning(str(value['id']) + ' failed mysql update')
                        logging.exception(e)

            con.commit()



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

if __name__ == '__main__':
    main()


rsp = rssProcessor()