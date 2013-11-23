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
import pandas as pd


def main():
    try:
        rawDat = pd.read_csv(sys.argv[1])
    except IndexError:
        print "\nAt the moment, this script needs a command line argument which is the name of the csv file you want to split\n"
        sys.exit()
    ds = dataSplitter()
    valDat = ds.validate(rawDat)
    splitDat = ds.split(valDat)
    for n in ['train','test']:
        dat = splitDat[n]
        ds.write(n,dat)


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
        valid = df.ix[keepList]
        return(valid.reset_index(drop=True))
    
    def split(self,df):
        lenDat = len(df)
        s = list(range(lenDat))
        random.shuffle(s)
        spl = int(math.floor(.75*lenDat))
        tr_ind = s[0:spl]
        te_ind = s[spl+1:]
        tr = df.ix[tr_ind]
        te = df.ix[te_ind]
        return ({'train': tr,'test':te})        
        
    def write(self,prefix,df):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%m-%d-%Y_%H-%M')
        cwd = os.getcwd()
        outFile = cwd + '/data/' + prefix + '/' + prefix + '_@'+st+'.csv'
        df.to_csv(outFile)
        print prefix + ' data successfully written ' + st

        
if __name__ == '__main__':
    main()