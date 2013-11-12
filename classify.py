#File: classify.py


'''what is happening?

1. Take files and create list of strings
2. Input this list of strings into the countvectorizer
3. Get the array
4. Train model

'''

from datetime import datetime
import numpy as np
import pylab as pl
from sklearn import linear_model

from sklearn.feature_extraction.text import CountVectorizer

from reporter import Reporter

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

