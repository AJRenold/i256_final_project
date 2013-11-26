#!/usr/bin/python

#File: classify.py

import os
import datetime
import time
import sys
import numpy as np
import pylab as pl
import pandas as pd
import DataPuller as DataPuller

from cPickle import load,dump
from sklearn import linear_model
from sklearn import metrics
from sklearn.utils.extmath import density


from sklearn.feature_extraction.text import CountVectorizer


def main():
    bm = Model('Basic')        
    sgdClas = linear_model.SGDClassifier(loss='log',penalty='elasticnet')
    benchmark(sgdClas,bm)
    
    
class Model:
    
    def __init__(self,modelType='Basic'):        
        if modelType == 'Basic':
            self.modelType = 'Basic'
            self.buildBasic()
        
    def buildBasic(self):
        tr_ind = self.getSetIndices('train')
        te_ind = self.getSetIndices('test')
        dp = DataPuller.DataPuller()
        rawdf = dp.fetchData()
        valList = dp.validate(rawdf,runReport=False)
        df = rawdf.ix[valList]
        self.sqlChecker(df,tr_ind,te_ind)
        self.train_df = df.ix[tr_ind]
        self.test_df = df.ix[te_ind]
        articleStrs_tr = self.train_df['reporter_content'].tolist()
        articleStrs_te = self.test_df['reporter_content'].tolist()
        self.CV = CountVectorizer()
        self.X_train = self.CV.fit_transform(articleStrs_tr)
        self.X_test = self.CV.transform(articleStrs_te)
        self.y_train = self.train_df['sensitive_flag'].tolist()
        self.y_test = self.test_df['sensitive_flag'].tolist()
        t = zip(self.CV.get_feature_names(),
        np.asarray(self.X_train.sum(axis=0)).ravel())
        self.freqDist =  sorted(t, key=lambda a: -a[1])
        self.feature_names = np.asarray(self.CV.get_feature_names())
                                
    def sqlChecker(self,df,pull_tr,pull_te):
        valrecs = len(df)
        #print 'valrecs: ' + str(valrecs)
        pickrecs = len(pull_tr) + len(pull_te)
        #print 'pickrecs: ' + str(pickrecs)
        if valrecs != pickrecs:
            print "The Database has been updated since your last pull. Do a git pull to get most recent version"
            sys.exit()
    
    def getSetIndices(self,type,getLatest=True,timestamp=''):
        if getLatest == True:
            cwd = os.getcwd()
            dirtoWalk = cwd + '/data/' + type
            files = os.listdir(dirtoWalk)
            try:
                files.remove('.placeholder')
                files.remove('.DS_Store')                
            except ValueError:
                pass          
            timeList= []
            for i in range(len(files)):
                f = files[i]
                #print f
                stampstr = f.split('@')[1].split('.')[0]
                ts = time.strptime(stampstr, "%m-%d-%Y_%H-%M")
                sec = time.mktime(ts)
                timeList.append(sec)
            latestTs = timeList.index(max(timeList))
            latestFi = files[latestTs]
            #data = pd.read_csv(dirtoWalk + '/' +latestFi)
            oP = open(dirtoWalk+'/'+latestFi,'r')
            indices = load(oP)
            return indices
            


    
    
def benchmark(clf,Model):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time.time()
    clf.fit(Model.X_train, Model.y_train)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time.time()
    pred = clf.predict(Model.X_test)
    test_time = time.time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(Model.y_test, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        
        #if opts.print_top10 and feature_names is not None:
        if Model.feature_names is not None:
            print("Top 10 tokens:")
            #for i, category in enumerate(categories):
            #   top10 = np.argsort(clf.coef_[i])[-10:]
            top10 = np.argsort(clf.coef_[0])[-10:]
            print(trim("%s"
                      % (" ".join(Model.feature_names[top10]))))
        print""

    #if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(Model.y_test, pred))
        #                                    target_names=categories))
    '''
    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))
    '''
    print ""
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time
    


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

        
if __name__ == '__main__':
   main()
