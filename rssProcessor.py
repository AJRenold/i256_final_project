
from bs4 import BeautifulSoup
import csv
import time
import datetime
import os

xmlFile ='nsfw.xml'

def main():
    rsp = rssProcessor()
    outDict = rsp.process(xmlFile)
    rsp.writeFile(outDict,xmlFile)

class rssProcessor:
    def __init__(self):
        self.isxmlProcessor = True

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
            outDict[link] = flag
        return outDict
    
    def writeFile(self,outDict,xmlFile):
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