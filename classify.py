#!/usr/bin/python

from collections import defaultdict
from copy import copy
import os
import datetime
import time
import string
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
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier


from sklearn import metrics
from sklearn.utils.extmath import density
from sklearn import cross_validation

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from nltk import (
        FreqDist,
        NaiveBayesClassifier,
        DecisionTreeClassifier,
        classify,
        word_tokenize,
        bigrams,
        precision,
        recall,
        f_measure
    )

from bigram_features import (
        bigrams_maximizing_prob_diff
    )

# parse commandline options and arguments
op = OptionParser()

op.add_option("--m","--model",dest='model_type',
              help="designate which model to use",action='store',type='string')

op.add_option("--r","--report",dest="training_document_info",
              help="report information on the training documents", action="store_true")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

# Check for a model type
if opts.model_type:
    MODEL = opts.model_type
else:
    MODEL = "Basic"

# Check for report action
if opts.training_document_info:
    REPORT_TRAINING_DOCS_INFO = opts.training_document_info
else:
    REPORT_TRAINING_DOCS_INFO = False

def main():
    bm = Model(MODEL, REPORT_TRAINING_DOCS_INFO)

    if 'NLTK' in MODEL:
        # For NLTK models, use benchmarkNLTK
        clf = DecisionTreeClassifier
        benchmarkNLTK(clf, bm)

        clf = NaiveBayesClassifier
        benchmarkNLTK(clf,bm)
    else:
        sgdClas = linear_model.SGDClassifier(loss='log',penalty='elasticnet')
        benchmarkSKLearn(sgdClas,bm)
        #crossValidate(sgdClas,bm)

        svmClas = svm.SVC(kernel='rbf')
        benchmarkSKLearn(svmClas, bm)
        #crossValidate(clas, bm)

class Model:

    def __init__(self,modelType='Basic',outputDocumentInfo=False):

        #Establish Core elements of any model
        train_idx = self.getSetIndices('train')
        test_idx = self.getSetIndices('test')

        # Use DataPuller to fetch and validate a dataframe from our database
        dp = DataPuller.DataPuller()
        rawdf = dp.fetchData()
        valList = dp.validate(rawdf,runReport=False)
        df = rawdf.ix[valList]

        ## Adds turk ratings to the dataframe
        df = self.addTurkRatingData(df)

        ## Adds 'content' column to the dataframe
        df = self.addContentColumns(df)

        # Check that the training and test dataframe indices are still valid
        # If this fails, run DataPuller.py to create new train and test data
        self.sqlChecker(df,train_idx,test_idx)
        self.train_df = df.ix[train_idx]
        self.test_df = df.ix[test_idx]

        ## filters the training dataframe to rows that have 3 or more turk ratings
        filter_len = self.train_df['count_turk_ratings'].map(lambda x: x >= 3)
        self.train_df = self.train_df[filter_len]

        ## filters the testing dataframe to rows that have 3 ore more turk ratings
        filter_len = self.test_df['count_turk_ratings'].map(lambda x: x >= 3)
        self.test_df = self.test_df[filter_len]

        articleStrs_tr = self.train_df['content'].tolist()
        articleStrs_te = self.test_df['content'].tolist()

        ## output corpus info if --r or --report
        if outputDocumentInfo:
            self.outputDocumentStats(articleStrs_tr, 'Training')
            self.outputDocumentStats(articleStrs_te, 'Testing')

        # Create the model features based on the --m or --model argument
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
            self.buildNLTKSingleWordFeatures(articleStrs_tr, articleStrs_te)

        if modelType == "NLTKBigram":
            self.modelType = "NLTKBigram"
            self.buildNLTKBigramFeatures(articleStrs_tr, articleStrs_te)

        if modelType == "NLTKBigramUnigramCombo":
            self.modelType = "NLTKBigramUnigramCombo"
            self.buildNLTKComboFeatures(articleStrs_tr, articleStrs_te)

    def buildBasic(self,articleStrs_tr,articleStrs_te):
        """
            Builds basic unigram features for sklearn classifiers
            Sets class attributes for Model use
        """
        self.CV = CountVectorizer(stop_words='english')
        self.X_train = self.CV.fit_transform(articleStrs_tr)
        self.X_test = self.CV.transform(articleStrs_te)
        self.y_train = self.train_df['turk_sensitive_flag'].tolist()
        self.y_test = self.test_df['turk_sensitive_flag'].tolist()
        t = zip(self.CV.get_feature_names(),
        np.asarray(self.X_train.sum(axis=0)).ravel())
        self.freqDist =  sorted(t, key=lambda a: -a[1])
        self.feature_names = np.asarray(self.CV.get_feature_names())

    def buildBasicBigram(self, articleStrs_tr, articleStrs_te):
        """
            Builds basic bigram features for sklearn classifiers
            Also can do combined unigram and bigram features if
            ngram_range is set to 1,2
            Sets class attributes for Model use
        """
        self.CV = CountVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b', stop_words='english',
                                )
        self.X_train = self.CV.fit_transform(articleStrs_tr)
        self.X_test = self.CV.transform(articleStrs_te)
        self.y_train = self.train_df['turk_sensitive_flag'].tolist()
        self.y_test = self.test_df['turk_sensitive_flag'].tolist()
        t = zip(self.CV.get_feature_names(),
        np.asarray(self.X_train.sum(axis=0)).ravel())
        self.freqDist =  sorted(t, key=lambda a: -a[1])
        self.feature_names = np.asarray(self.CV.get_feature_names())

    def buildInvDoc(self,articleStrs_tr,articleStrs_te):
        """
            Builds features using TfidfVectorizer for sklearn classifiers
            Sets class attributes for Model use
        """
        self.TFV = TfidfVectorizer()
        self.X_train = self.TFV.fit_transform(articleStrs_tr)
        self.X_test = self.TFV.transform(articleStrs_te)
        self.y_train = self.train_df['turk_sensitive_flag'].tolist()
        self.y_test = self.test_df['turk_sensitive_flag'].tolist()
        t = zip(self.TFV.get_feature_names(),
        np.asarray(self.X_train.sum(axis=0)).ravel())
        self.freqDist =  sorted(t, key=lambda a: -a[1])
        self.feature_names = np.asarray(self.TFV.get_feature_names())

    def buildNLTKSingleWordFeatures(self, articleStrs_tr, articleStrs_te):
        """
            Builds unigram features for use with NLTK classifiers
            Sets class attributes for Model use
        """
        def wordFeatures(doc):
            features = {}
            for word in re.sub('\W',' ',doc).lower().split(' '):
                features[word] = True

            return features

        training_features = [ wordFeatures(doc) for doc in articleStrs_tr ]
        testing_features = [ wordFeatures(doc) for doc in articleStrs_te ]

        self.training_set = zip(training_features, self.train_df['turk_sensitive_flag'].tolist())
        self.test_set = zip(testing_features, self.test_df['turk_sensitive_flag'].tolist() )
    
    def buildNLTKBigramFeatures(self, articleStrs_tr, articleStrs_te):
        """
            Builds bigram features for use with NLTK classifiers
            Sets class attributes for Model use
        """

        def bigramFeatures(doc, feature_set):
            features = copy(feature_set)
            words = [w.lower() for w in word_tokenize(doc) if w not in string.punctuation]
            for bigram in bigrams(words):
                if bigram in feature_set:
                    features[bigram] = True

            return features

        bigram_features = { bigram: False for prob_diff, bigram in
                                bigrams_maximizing_prob_diff(zip(articleStrs_tr, 
                                self.train_df['turk_sensitive_flag'].tolist()), 1000) }

        training_features = [ bigramFeatures(doc, bigram_features) for doc in articleStrs_tr ]
        testing_features = [ bigramFeatures(doc, bigram_features) for doc in articleStrs_te ]

        self.training_set = zip(training_features, self.train_df['turk_sensitive_flag'].tolist())
        self.test_set = zip(testing_features, self.test_df['turk_sensitive_flag'].tolist() )

    def buildNLTKComboFeatures(self, articleStrs_tr, articleStrs_te):
        """
            Builds unigram and bigram features for NLTK classifiers
            Sets class attributes for Model use
        """

        def bigramFeatures(doc, feature_set):
            features = copy(feature_set)
            words = [w.lower() for w in word_tokenize(doc) if w not in string.punctuation]
            for bigram in bigrams(words):
                if bigram in feature_set:
                    features[bigram] = True

            return features

        def wordFeatures(doc):
            features = {}
            for word in re.sub('\W',' ',doc).lower().split(' '):
                features[word] = True

            return features

        bigram_features = { bigram: False for prob_diff, bigram in
                                bigrams_maximizing_prob_diff(zip(articleStrs_tr, 
                                self.train_df['turk_sensitive_flag'].tolist()), 1000) }

        def createFeatureList(docs):
            feature_list = []
            for doc in docs:
                feat = {}
                feat.update(wordFeatures(doc))
                feat.update(bigramFeatures(doc, bigram_features))
                feature_list.append(feat)

            return feature_list

        training_features = createFeatureList(articleStrs_tr)
        testing_features = createFeatureList(articleStrs_te)

        self.training_set = zip(training_features, self.train_df['turk_sensitive_flag'].tolist())
        self.test_set = zip(testing_features, self.test_df['turk_sensitive_flag'].tolist() )

    def addContentColumns(self, df):
        """
            Adds content and content_len columns
            to dataframe based on parser_content column
        """

        def getText(parser_content):
            soup = BeautifulSoup(parser_content)
            text = re.sub('\n',' ',soup.getText().strip())
            return text

        def getTextLen(content):
            return len(content.split(' '))

        df['content'] = df['parser_content'].apply(getText)
        df['content_len'] = df['content'].apply(getTextLen)

        return df

    def addTurkRatingData(self, df):
        """
            Add turker rating columns 'count_turk_ratings',
            'avg_turk_ratings', and 'turk_sensitive_flag'
            to dataframe based the turk_rating columns
        """
        def countTurkRatings(row):
            count = 0
            for i in range(5):
                if row['turk_rating'+str(i)] != '':
                    count += 1

            return count

        def averageTurkRating(row):
            ratings = []
            for i in range(5):
                if row['turk_rating'+str(i)] != '':
                    ratings.append(int(row['turk_rating'+str(i)]))

            return np.mean(ratings)

        def turkRatingLabel(avg_rating):
            if avg_rating >= 0.5:
                return 1
            else:
                return 0

        df['count_turk_ratings'] = df.apply(countTurkRatings, axis=1)
        df['avg_turk_ratings'] = df.apply(averageTurkRating, axis=1)
        df['turk_sensitive_flag'] = df['avg_turk_ratings'].apply(turkRatingLabel)

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
        """
            Checks that MySQL data has not been updated
            If it has, DataPuller.py needs to be run again
            because the pull_tr and pull_te indices do not
            match the dataframe
        """
        valrecs = len(df)
        #print 'valrecs: ' + str(valrecs)
        pickrecs = len(pull_tr) + len(pull_te)
        #print 'pickrecs: ' + str(pickrecs)
        if valrecs != pickrecs:
            print "The Database has been updated since your last pull. Do a git pull to get most recent version"
            sys.exit()
    
    def getSetIndices(self,type,getLatest=True,timestamp=''):
        """
            From the /data/ directory, finds the most recently
            created test and training indices
        """
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
    """
        Benchmark nltk classifiers and output report
    """
    print('_' * 80)
    print("Training: [Model Type: "+ MODEL+ "]")

    print(classifier)
    t0 = time.time()
    clf = classifier.train(Model.training_set)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)

    refsets = defaultdict(set)
    testsets = defaultdict(set)

    t0 = time.time()
    for i, (features, label) in enumerate(Model.test_set):
        refsets[label].add(i)
        observed = clf.classify(features)
        testsets[observed].add(i)

    test_time = time.time() - t0
    print("test time:  %0.3fs" % test_time)
    print 'Sensitive precision:', precision(refsets[1], testsets[1])
    print 'Sensitive recall:', recall(refsets[1], testsets[1])
    print 'Sensitive F-measure:', f_measure(refsets[1], testsets[1])
    print 'Not Sens precision:', precision(refsets[0], testsets[0])
    print 'Not Sens recall:', recall(refsets[0], testsets[0])
    print 'Not Sens F-measure:', f_measure(refsets[0], testsets[0])

    print("\n50 most informative features")
    clf.show_most_informative_features(50)

def prepROC(trainedClas,Model):
    """
        Prepate a csv data file for the ROC plot
    """
    df_probs = pd.DataFrame(trainedClas.predict_proba(Model.X_test))
    df_probs.to_csv('pd_'+MODEL+'_proba'+'.csv')
    print('Please add name of classifier to file name')

def crossValidate(clf,Model):
    """
        Cross Validation for sklearn classifiers
    """
    print("Cross Validation: [Model Type: "+ MODEL+ "]")
    scores = cross_validation.cross_val_score(clf, Model.X_train, np.array(Model.y_train), 
                                        cv=10, scoring='f1')
    print('\n')
    print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print('\n')

def benchmarkSKLearn(clf,Model):
    """
        Benchmark sklearn classifiers and output report
    """
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

        if Model.feature_names is not None:
            print("Top 10 tokens:")
            top10 = np.argsort(clf.coef_[0])[-10:]
            print(trim("%s"
                      % (" ".join(Model.feature_names[top10]))))
            print("\nNext 10")
            top10to20 = np.argsort(clf.coef_[0])[-20:-10]
            print(trim("%s"
                      % (" ".join(Model.feature_names[top10to20]))))
            print("\nNext 10")
            top20to30 = np.argsort(clf.coef_[0])[-30:-20]
            print(trim("%s"
                      % (" ".join(Model.feature_names[top20to30]))))
        print

        print("classification report:")
        print(metrics.classification_report(Model.y_test, pred))

    print ""
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

if __name__ == '__main__':
   main()
