import re
import nltk

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

print("Sanity Check!")

#load swedish corpus
with open('LAS2_corpus.txt') as my_file:
    corpus_s = my_file.read().splitlines()
corpus_s = [re.sub(r'[^\w\s]', '', re.sub(r'[0-9]', '', s)) for s in corpus_s]

corpus = " ".join(corpus_s)
words = nltk.word_tokenize(corpus)
word_frequency = nltk.FreqDist(words)

tfIdfVectorizer=TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(corpus_s[:2000]).todense() #take first 2000 sentences
#0.49
tfIdf[tfIdf< 0.28] = np.nan

#1) Length of sentences 
def length_s (s):
    return len(nltk.word_tokenize(s))
    # return [len(nltk.word_tokenize(s)) for s in sentences]
#2) Mean length of words 
def length_w (s):
    return np.mean( [len(w) for w in nltk.word_tokenize(s)] )

#3) types (is not normalized)
def types (s):
    return len(set(nltk.word_tokenize(s)))


#4) TTR and its transformations (Scaled version of TTR) are dependent on the length of the sentences N
#TTR = T/N
def TTR(s):
    return len(set(nltk.word_tokenize(s))) / len(nltk.word_tokenize(s))

#root TTR = T/sqrt(N)
def rootTTR(s):
    return len(set(nltk.word_tokenize(s))) / len(nltk.word_tokenize(s))**0.5

#corrected TTR = T/sqrt(2N)
def correctedTTR(s):
    return len(set(nltk.word_tokenize(s))) / (2*len(nltk.word_tokenize(s)))**0.5

#log TTR = log(T)/log(N)
def logTTR(s):
    return np.log(len(set(nltk.word_tokenize(s)))) / np.log(len(nltk.word_tokenize(s)))



#5) OVIX
def ovix(s):
    return np.log(len(nltk.word_tokenize(s))) / (2- (np.log(len(set(nltk.word_tokenize(s))) )/np.log(len(nltk.word_tokenize(s)))) )



#6 TF-IDF
def get_mean_tfidf_indices(thresh_top=1115):
    indices = {}
    # tfIdfVectorizer=TfidfVectorizer(use_idf=True)
    # tfIdf = tfIdfVectorizer.fit_transform(corpus_s[:2000]).todense() #take first 2000 sentences
    # #0.49
    # tfIdf[tfIdf< 0.28] = np.nan
    
    df = pd.DataFrame(np.nanmean(tfIdf.T, axis=1), index=tfIdfVectorizer.get_feature_names_out(), columns=['mean'])

    indices = list(df.sort_values('mean', ascending=False).index) [:thresh_top]
    
    return indices

def tfidf (s, thresh_top=1115):
    indices_mean = get_mean_tfidf_indices(thresh_top)
    return len([w  for w in nltk.word_tokenize(s) if w in indices_mean]) / len(nltk.word_tokenize(s))


#7) Lexical Profile 
def mostfrequentwords(s, top_freq =60):
    # corpus = " ".join(corpus_s)
    # words = nltk.word_tokenize(corpus)
    # word_frequency = nltk.FreqDist(words)
    word_mostcommon = [t[0] for t in word_frequency.most_common()[:top_freq]]
    return len([w  for w in nltk.word_tokenize(s) if w in word_mostcommon]) / len(nltk.word_tokenize(s))

def leastfrequentwords(s,top_freq=24000):
    # corpus = " ".join(corpus_s)
    # words = nltk.word_tokenize(corpus)
    # word_frequency = nltk.FreqDist(words)
    word_leastcommon = [t[0] for t in word_frequency.most_common()[-top_freq: ] ]
#     word_leastcommon = [t[0] for t in word_frequency.most_common() if t[1]==1 ]
    return len([w  for w in nltk.word_tokenize(s) if w in word_leastcommon]) / len(nltk.word_tokenize(s))
