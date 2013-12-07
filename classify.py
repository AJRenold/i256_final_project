#!/usr/bin/python

#File: classify.py

import os
import datetime
import time
import sys
import re

from bs4 import BeautifulSoup
import numpy as np
import pylab as pl
import pandas as pd
import DataPuller as DataPuller

from optparse import OptionParser
from cPickle import load,dump
from sklearn import linear_model
from sklearn import metrics
from sklearn.utils.extmath import density

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from nltk import (
        FreqDist,
        NaiveBayesClassifier,
        classify
    )

# parse commandline options and arguments
op = OptionParser()
'''op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")'''

op.add_option("--m","--model",dest='model_type',
              help="designate which model to use",action='store',type='string')

op.add_option("--r","--report",dest="training_document_info",
              help="report information on the training documents", action="store_true")


(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

'''print(__doc__)
op.print_help()
print()'''

if opts.model_type:
    MODEL = opts.model_type
else:
    MODEL = "Basic"

if opts.training_document_info:
    REPORT_TRAINING_DOCS_INFO = opts.training_document_info
else:
    REPORT_TRAINING_DOCS_INFO = False

def main():
    bm = Model(MODEL, REPORT_TRAINING_DOCS_INFO)
    
    if 'NLTK' in MODEL:
        clf = NaiveBayesClassifier
        benchmarkNLTK(clf,bm)
    else:
        sgdClas = linear_model.SGDClassifier(loss='log',penalty='elasticnet')
        benchmarkSKLearn(sgdClas,bm)

class Model:
    
    def __init__(self,modelType='Basic',trainingDocumentInfo=False):
        
        #Establish Core elements of any model
        tr_ind = self.getSetIndices('train')
        te_ind = self.getSetIndices('test')

        dp = DataPuller.DataPuller()
        rawdf = dp.fetchData()
        valList = dp.validate(rawdf,runReport=False)
        df = rawdf.ix[valList]
        self.sqlChecker(df,tr_ind,te_ind)
        self.train_df = df.ix[tr_ind]
        self.test_df = df.ix[te_ind]

        self.train_df = self.addContentColumns(self.train_df)
        self.test_df = self.addContentColumns(self.test_df)

        articleStrs_tr = self.train_df['content'].tolist()
        articleStrs_te = self.test_df['content'].tolist()

        if trainingDocumentInfo:
            self.outputDocumentStats(articleStrs_tr, 'Training')

        if trainingDocumentInfo:
            self.outputDocumentStats(articleStrs_te, 'Testing')

        #Put specific model code here:
        if modelType == 'Basic':
            self.modelType = 'Basic'
            self.buildBasic(articleStrs_tr,articleStrs_te)
        
        if modelType == "InvDocFreq":
            self.modelType = "InvDocFreq"
            self.buildInvDoc(articleStrs_tr,articleStrs_te)

        if modelType == "BasicBigram":
            self.modelType = "BasicBigram"
            self.buildBasicBigram(articleStrs_tr, articleStrs_te)

        if modelType == "NLTKSingleWord":
            self.modelType = "NLTKSingleWord"
            self.buldNLTKSingleWordFeatures(articleStrs_tr, articleStrs_te)

    def buildBasic(self,articleStrs_tr,articleStrs_te):
        self.CV = CountVectorizer()
        self.X_train = self.CV.fit_transform(articleStrs_tr)
        self.X_test = self.CV.transform(articleStrs_te)
        self.y_train = self.train_df['sensitive_flag'].tolist()
        self.y_test = self.test_df['sensitive_flag'].tolist()
        t = zip(self.CV.get_feature_names(),
        np.asarray(self.X_train.sum(axis=0)).ravel())
        self.freqDist =  sorted(t, key=lambda a: -a[1])
        self.feature_names = np.asarray(self.CV.get_feature_names())

    def buildBasicBigram(self, articleStrs_tr, articleStrs_te):
        self.CV = CountVectorizer(ngram_range=(2,2), token_pattern=r'\b\w+\b')
        self.X_train = self.CV.fit_transform(articleStrs_tr)
        self.X_test = self.CV.transform(articleStrs_te)
        self.y_train = self.train_df['sensitive_flag'].tolist()
        self.y_test = self.test_df['sensitive_flag'].tolist()
        t = zip(self.CV.get_feature_names(),
        np.asarray(self.X_train.sum(axis=0)).ravel())
        self.freqDist =  sorted(t, key=lambda a: -a[1])
        self.feature_names = np.asarray(self.CV.get_feature_names())

    def buildInvDoc(self,articleStrs_tr,articleStrs_te):
        self.TFV = TfidfVectorizer()
        self.X_train = self.TFV.fit_transform(articleStrs_tr)
        self.X_test = self.TFV.transform(articleStrs_te)
        self.y_train = self.train_df['sensitive_flag'].tolist()
        self.y_test = self.test_df['sensitive_flag'].tolist()
        t = zip(self.TFV.get_feature_names(),
        np.asarray(self.X_train.sum(axis=0)).ravel())
        self.freqDist =  sorted(t, key=lambda a: -a[1])
        self.feature_names = np.asarray(self.TFV.get_feature_names())

    def buldNLTKSingleWordFeatures(self, articleStrs_tr, articleStrs_te):
        
        def wordFeatures(doc):
            features = {}
            for word in re.sub('\W',' ',doc).lower().split(' '):
                features[word] = True

            return features

        training_features = [ wordFeatures(doc) for doc in articleStrs_tr ]
        testing_features = [ wordFeatures(doc) for doc in articleStrs_te ]

        self.training_set = zip(training_features, self.train_df['sensitive_flag'].tolist())
        self.test_set = zip(testing_features, self.test_df['sensitive_flag'].tolist() )


    def addContentColumns(self, df):

        def getText(parser_content):
            soup = BeautifulSoup(parser_content)
            text = re.sub('\n',' ',soup.getText().strip())
            return text

        def getTextLen(content):
            return len(content.split(' '))

        df['content'] = df['parser_content'].apply(getText)
        df['content_len'] = df['content'].apply(getTextLen)

        return df

    def outputDocumentStats(self, documents, doc_set_name):
        """
            Outputs a information on document list
        """
        print('_' * 80)
        print('%s Documents Statistics' % doc_set_name)

        print('number of documents: %d' % len(documents))

        corpus_words = [ word for doc in documents
                        for word in re.sub('\W',' ',doc).lower().split(' ') ]

        avg_doc_length = len(corpus_words) / float(len(documents))
        print('average document length: %0.3f words' % avg_doc_length )
        print('approx. number of words in corpus: %d' % len(corpus_words))
        vocab = set(corpus_words)
        print('approx. vocabulary size: %d' % len(vocab))

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

def benchmarkNLTK(classifier,Model):
    print('_' * 80)
    print("Training: [Model Type: "+ MODEL+ "]")

    print(classifier)
    t0 = time.time()
    clf = classifier.train(Model.training_set)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time.time()
    accuracy = classify.accuracy(clf, Model.test_set)
    test_time = time.time() - t0
    print("test time:  %0.3fs" % test_time)
    print("nltk classify accuracy:   %0.3f" % accuracy)

    print("50 most informative features")
    clf.show_most_informative_features(50)

def benchmarkSKLearn(clf,Model):
    print('_' * 80)
    
    print("Training: [Model Type: "+ MODEL+ "]")
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
