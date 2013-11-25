#!/usr/bin/python

'''Accepts file name of csv document command line argument, 
   outputs training and test sets into data directory with 
   time stamp'''


import datetime
import time
import math
import random
import os
import sys
from cPickle import dump,load
import pandas as pd

#testingDat = '/Users/dgreis/Documents/School/Classes/INFO_256/Outbrain/i256_final_project/data/forTesting/buzzLinkTest/RSS.csv'

def main():
    try:
        rawDat = pd.read_csv(sys.argv[1])
        #rawDat = pd.read_csv(testingDat)
    except IndexError:
        print "\nAt the moment, this script needs a command line argument which is the name of the csv file you want to split\n"
        sys.exit()
    ds = dataSplitter()
    valList = ds.validate(rawDat)
    splitDict = ds.split(valList)
    for n in ['train','test']:
        ds.write(n,splitDict)


class dataSplitter:

    def validate(self,df):
        series = df['reporter_content']
        probList = []
        keepList = []
        for i in range(len(series)):
            ent = series[i]
           # print ent
            if isinstance(ent, float) == True:
                probList.append(df['id'][i])
            elif ent == 'None':
                probList.append(df['id'][i])
            else:
                keepList.append(i)
        if len(probList) > 0:
            print "\nATTENTION: Problem with 'reporter_content' cell in rows with following id's: "
            for p in probList:
                print str(p)
            print('*'*40)
        #valid = df.ix[keepList]
        #return(valid.reset_index(drop=True))
        return keepList
    
    
    def split(self,keepList):
        lenDat = len(keepList)
        s = list(range(lenDat))
        random.shuffle(s)
        spl = int(math.floor(.75*lenDat))
        randInd_tr = s[0:spl]
        randInd_te = s[spl+1:]
        #tr = df.ix[tr_ind]
        #te = df.ix[te_ind]
        tr = [keepList[i] for i in randInd_tr]
        te = [keepList[i] for i in randInd_te]
        return ({'train': tr,'test':te})        
        
    def write(self,prefix,outDict):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%m-%d-%Y_%H-%M')
        cwd = os.getcwd()
        outName = cwd + '/data/' + prefix + '/' + prefix + '_@'+st+'.pk1'
        '''with open(outFile, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow((prefix,'rows'))
            for k in outDict:
                writer.writerow((k,outDict[k]))
        csvfile.close()
        df.to_csv(outFile)
        outName = cwd+'/allInputFiles.pk1'''
        outFile = open(outName,'wb')
        dump(outDict[prefix],outFile)
        print prefix + ' data index successfully written ' + st

        
if __name__ == '__main__':
    main()