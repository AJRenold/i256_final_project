#!/usr/bin/python

from cPickle import dump,load
import datetime
import math
import os
import pandas as pd
import random
import sys
import time

class dataSplitter:

    def split(self,keepList):
        """
            Given a list of dataframe indices, randomly
            shuffle the list and then split it .75 for training
            and .25 for testing
        """
        lenDat = len(keepList)
        s = list(range(lenDat))
        random.shuffle(s)
        spl = int(math.floor(.75*lenDat))
        randInd_tr = s[0:spl]
        randInd_te = s[spl:]
        tr = [keepList[i] for i in randInd_tr]
        te = [keepList[i] for i in randInd_te]
        return ({'train': tr,'test':te})        

    def write(self,prefix,outDict):
        """
            Writes out a pickle and creates a filename 
            that ends with the current date. The filename is
            important so that we always train and test
            on the most recently written file
        """
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%m-%d-%Y_%H-%M')
        cwd = os.getcwd()
        outName = cwd + '/data/' + prefix + '/' + prefix + '_@'+st+'.pk1'
        with open(outName, 'wb') as outFile:
            dump(outDict[prefix],outFile)
            print prefix + ' data index successfully written ' + st
