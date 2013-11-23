#!/usr/bin/python

'''Accepts file name of csv document command line argument, 
   outputs training and test sets into data directory with 
   time stamp'''


import datetime
import time
import math
import os
import sys
import pandas as pd


def main():
    rawDat = pd.read_csv(sys.argv[1])
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
            if isinstance(ent, float) == True:
                probList.append(i)
            else:
                keepList.append(i)
        print("Problem with 'reporter_content' cell in following rows: " + ''.join(str(probList)))
        return(df.ix[keepList])
    
    def split(self,df):
        lenDat = len(data)
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
        st = datetime.datetime.fromtimestamp(ts).strftime('%m-%d-%Y_%H_%M')
        cwd = os.getcwd()
        outFile = cwd + '/data/' + prefix + '/' + prefix + '_'+st+'.csv'
        df.to_csv(outFile)
        print prefix + ' data successfully written ' + st
        

        
if __name__ == '__main__':
    main()