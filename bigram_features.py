import nltk
from nltk.tokenize import word_tokenize
#from nltk.collocations import BigramCollocationFinder
#from nltk.metrics import BigramAssocMeasures
import string

def bigrams_maximizing_prob_diff(docs, n, min_freq=3, min_word_len=3, stopwords=[]):

    def get_max_diff_bigrams(bigrams, pos, neg):
        prob_diff_bigrams = []
        pos_fd = pos.freqdist()
        neg_fd = neg.freqdist()

        # For each bigram, find the positive and negative probabilities
        for bigram in bigrams:
            p = pos.prob(bigram)
            n = neg.prob(bigram)

            # Check if the frequency of occurrence of the bigram in either positive or negative sentences is > 15
            if pos_fd[bigram] >= min_freq or neg_fd[bigram] >= min_freq:
                # Find the absolute difference in probability for each bigram
                prob_diff_bigrams.append((abs(p - n), bigram))

        # Return the bigram list sorted based on the absolute difference between positive and negative probabilities.
        return sorted(prob_diff_bigrams, reverse=True)

    # Get all the bigrams
    labelled_bigrams = get_bigrams(docs)

    # Calculate the conditional frequency distribution for positive and negative bigrams.
    cfd = nltk.ConditionalFreqDist((label, bigram)
                                        for label, bigrams in labelled_bigrams
                                        for bigram in bigrams
                                        if (len(bigram[0]) > min_word_len or len(bigram[1]) > min_word_len))

    # Calculate the conditional probability distribution for the computed conditional frequency distribution
    cpdist = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
    pos_dist = cpdist[1]
    neg_dist = cpdist[0]

    bigrams_samples = list(set(pos_dist.samples()).union(set(neg_dist.samples())))

    return get_max_diff_bigrams(bigrams_samples, pos_dist, neg_dist)[:n]

def get_bigrams(docs):
    bigrams = []
    for doc, label in docs:
        words = [w.lower() for w in word_tokenize(doc) if w not in string.punctuation]
        bigrams.append((label, nltk.bigrams(words)))

    return bigrams
