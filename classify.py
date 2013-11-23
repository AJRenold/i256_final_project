#!/usr/bin/python

#File: classify.py


'''what is happening?

1. Take files and create list of strings
2. Input this list of strings into the countvectorizer
3. Get the array
4. Train model

'''

import os
import datetime
import time
import numpy as np
import pylab as pl
import pandas as pd
from sklearn import linear_model

from sklearn.feature_extraction.text import CountVectorizer

#Dummy code
r = Reporter()
articleText = []
r.read(url='http://espn.go.com/nba/story/_/id/9961147/sources-nba-tries-end-costly-aba-deal')
articleText.append(r.report_news())
r.read(url='http://www.forbes.com/sites/christopherhelman/2013/11/11/attention-fracktivists-corn-ethanol-is-the-real-environmental-culprit/')
articleText.append(r.report_news())

#Model code
cv = CountVectorizer()
X = cv.fit_transform(articleText)
t = zip(cv.get_feature_names(),
    np.asarray(cvfit.sum(axis=0)).ravel())
sort =  sorted(t, key=lambda a: -a[1])
sgdClas = linear_model.SGDClassifier(loss='log',penalty='elasticnet')

def fetchData(type,getLatest=True,timestamp=''):
    if getLatest == True:
        cwd = os.getcwd()
        dirtoWalk = cwd + '/data/' + type
        files = os.listdir(dirtoWalk)
        try:
            files.remove('.DS_Store')
        except ValueError:
            pass          
        timeList= []
        for i in range(len(files)):
            f = files[i]
            stampstr = f.split('@')[1].split('.')[0]
            ts = time.strptime(stampstr, "%m-%d-%Y_%H-%M")
            sec = time.mktime(ts)
            timeList.append(sec)
        latestTs = timeList.index(max(timeList))
        latestFi = files[latestTs]
        data = pd.read_csv(dirtoWalk + '/' +latestFi)
        return data

