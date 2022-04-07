import re
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

def corpus_loader(LANG):
    if LANG == 'fi':
        with open('scripts/LAS2_corpus.txt') as my_file:
            corpus_s = my_file.read().splitlines()
        corpus_s = [re.sub(r'[^\w\s]', '', re.sub(r'[0-9]', '', s)) for s in corpus_s]
    if LANG == 'sv':
        with open('scripts/lasbart.txt') as my_file:
            corpus_s = my_file.read().splitlines()
        corpus_s = [re.sub(r'[^\w\s]', '', re.sub(r'[0-9]', '', s)) for s in corpus_s]
    return corpus_s


def generate_lexical_features(transcript, LANG):

    corpus_s = corpus_loader(LANG)

    corpus = " ".join(corpus_s)
    words = nltk.word_tokenize(corpus)
    word_frequency = nltk.FreqDist(words)

    tfIdfVectorizer=TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(corpus_s[:2000]).todense() #take first 2000 sentences
    #0.49
    if LANG == 'fi':
        tfIdf[tfIdf< 0.49] = np.nan
        top_tfidf_thresh = 1115
        top_most_freq = 60
        top_least_freq = 24000
    if LANG  == 'sv':
        tfIdf[tfIdf< 0.28] = np.nan
        top_tfidf_thresh = 1450
        top_most_freq = 10500
        top_least_freq = 24000
    
    feature_dict = dict()

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
    def get_mean_tfidf_indices(thresh_top = top_tfidf_thresh):
        indices = {}        
        df = pd.DataFrame(np.nanmean(tfIdf.T, axis=1), index=tfIdfVectorizer.get_feature_names_out(), columns=['mean'])
        indices = list(df.sort_values('mean', ascending=False).index) [:thresh_top]
        return indices

    def tfidf (s, thresh_top = top_tfidf_thresh):
        indices_mean = get_mean_tfidf_indices(thresh_top)
        return len([w  for w in nltk.word_tokenize(s) if w in indices_mean]) / len(nltk.word_tokenize(s))


    #7) Lexical Profile 
    def mostfrequentwords(s, top_freq = top_most_freq):
        word_mostcommon = [t[0] for t in word_frequency.most_common()[:top_freq]]
        return len([w  for w in nltk.word_tokenize(s) if w in word_mostcommon]) / len(nltk.word_tokenize(s))

    def leastfrequentwords(s,top_freq  = top_least_freq):
        word_leastcommon = [t[0] for t in word_frequency.most_common()[-top_freq: ] ]
        return len([w  for w in nltk.word_tokenize(s) if w in word_leastcommon]) / len(nltk.word_tokenize(s))

    feature_dict['length_s'] = length_s(transcript)
    feature_dict['length_w'] = length_w(transcript)
    feature_dict['types'] = types(transcript)
    feature_dict['TTR'] = TTR(transcript)
    feature_dict['rootTTR'] = rootTTR(transcript)
    feature_dict['correctedTTR'] = correctedTTR(transcript)
    feature_dict['logTTR'] = logTTR(transcript)
    feature_dict['ovix'] = ovix(transcript)
    feature_dict['tfidf'] = tfidf(transcript)
    feature_dict['mostfrequentwords'] = mostfrequentwords(transcript)
    feature_dict['leastfrequentwords'] = leastfrequentwords(transcript)

    return feature_dict



# print(generate_lexical_features("Jag skulle vilja ha lite mjölk.", 'sv'))

# print(generate_lexical_features("Haluaisin vähän maitoa.", 'fi'))
