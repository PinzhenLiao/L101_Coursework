
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import pandas as pd
import ftfy
import matplotlib.pyplot as plt
import seaborn
import pycountry
import nltk 
import sys
import glob
import scipy
import re
import math
import string
import os
from nltk.tokenize import RegexpTokenizer
from math import log
import random
import statistics
from statistics import pvariance
from statistics import mean
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()

# Data Collection & Data Containers
import sqlite3
from collections import Counter, defaultdict

# Data Cleanup & Exploratory Data Analysis
# import email  - didn't work on raw email txt
import re
import numpy as np
import pandas as pd
import ftfy
import pycountry


# Modeling: Part 1
from textblob import TextBlob
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
# Modeling: Part 2
import gensim


# Visualization
import matplotlib.pyplot as plt
import seaborn
from wordcloud import WordCloud
#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD 
from sklearn.preprocessing import normalize 
from helpers import * 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD 
from sklearn.preprocessing import normalize 

#from helpers import * 


# 
# # Topic extraction with Non-negative Matrix Factorization and Latent Dirichlet Allocation
# 
# 
# This is an example of applying Non-negative Matrix Factorization
# and Latent Dirichlet Allocation on a corpus of documents and
# extract additive models of the topic structure of the corpus.
# The output is a list of topics, each represented as a list of terms
# (weights are not shown).
# 
# The default parameters (n_samples / n_features / n_topics) should make
# the example runnable in a couple of tens of seconds. You can try to
# increase the dimensions of the problem, but be aware that the time
# complexity is polynomial in NMF. In LDA, the time complexity is
# proportional to (n_samples * iterations).
# 

# In[3]:




# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups



def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.
tokenizer = RegexpTokenizer(r'\w+')
data_samples = []
stemmer = PorterStemmer()
print("Loading dataset...")
t0 = time()
dataset = fetch_20newsgroups(shuffle=True, random_state=1,                             remove=('headers', 'footers', 'quotes'))

print("done in %0.3fs." % (time() - t0))
for i,news in enumerate(dataset.data):
    tokenized = tokenizer.tokenize(news)
    tokenized = [word for word in tokenized if (nltk.pos_tag([word])[0][1] != 'JJ'                                                 and nltk.pos_tag([word])[0][1] != 'CC'                                                and nltk.pos_tag([word])[0][1] != 'CD'                                                and nltk.pos_tag([word])[0][1] != 'RB'                                                and nltk.pos_tag([word])[0][1] != 'DT'                                                and nltk.pos_tag([word])[0][1] != 'IN'                                                and nltk.pos_tag([word])[0][1] != 'PP')] 
    filtered = [stemmer.stem(word) for word in tokenized]
    filtered_1 = [words_1 for words_1 in filtered if re.search('[a-zA-z]',words_1)]
    filtered_2 = [words_2 for words_2 in filtered_1 if len(words_2)>2]
    stemmed_file = ' '.join(filtered_2)
    stemmed_file_1 = re.sub(r"[0-9]+[a-z]+","",stemmed_file)
    data_samples.append(stemmed_file_1)
    print(i, end='\r')


# In[5]:


def topic_TopWords(model, feature_names, n_top_words):
    """
    Function for printing % words contained by topic, n_top_words sorted by length,
    and plots top 10 words by importance per topic
    """
    for topic_idx, topic in enumerate(model.components_):

        print('Topic', (topic_idx+1))
        print("Percentage of Words", np.count_nonzero(topic))
        top_words = [feature_names[i] for i in topic.argsort()[::-1][:n_top_words]]
        print(' '.join(sorted(top_words,key=len,reverse=True)))
        top10_idx = topic.argsort()[::-1][:10]
        value = sorted(topic[top10_idx],reverse=False)

        # Horizontal Bar Plots of Top 10 weighted terms
        plt.figure(topic_idx + 1)
        plt.barh(np.arange(10) + .5, value, color="green", align="center")
        plt.yticks(np.arange(10) + .5, [feature_names[i] for i in topic.argsort()[::-1][:10]])
        plt.xlabel("Weight")
        plt.ylabel("Term")
        plt.title("Top 10 Highest Weighted Terms in Topic {}".format(topic_idx + 1))
        plt.grid(True)
        plt.show()



# In[6]:


n_samples = 2000
n_topics = 10
n_top_words = 20
n_features = 10000

def print_top_words(model, feature_names,n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()



# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   stop_words='english',ngram_range=(1, 2))
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                stop_words='english',ngram_range=(1, 2))
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)
topic_TopWords(nmf, tfidf_feature_names, 20)
print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
topic_TopWords(lda, tf_feature_names, 20)

svd = TruncatedSVD(n_components=n_topics)
t0 = time()
svd.fit(tfidf)  

print_top_words(svd, tfidf_feature_names, n_top_words)
topic_TopWords(svd, tfidf_feature_names, 20)


# In[7]:


n_samples = 2000
n_topics = 15
n_top_words = 20
n_features = 10000

def print_top_words(model, feature_names,n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()



# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   stop_words='english',ngram_range=(1, 2))
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                stop_words='english',ngram_range=(1, 2))
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)
topic_TopWords(nmf, tfidf_feature_names, 20)
print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
topic_TopWords(lda, tf_feature_names, 20)

svd = TruncatedSVD(n_components=n_topics)
t0 = time()
svd.fit(tfidf)  

print_top_words(svd, tfidf_feature_names, n_top_words)
topic_TopWords(svd, tfidf_feature_names, 20)


# In[8]:


n_samples = 2000
n_topics = 20
n_top_words = 20
n_features = 10000

def print_top_words(model, feature_names,n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()



# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   stop_words='english',ngram_range=(1, 2))
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                stop_words='english',ngram_range=(1, 2))
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)
topic_TopWords(nmf, tfidf_feature_names, 20)
print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
topic_TopWords(lda, tf_feature_names, 20)

svd = TruncatedSVD(n_components=n_topics)
t0 = time()
svd.fit(tfidf)  

print_top_words(svd, tfidf_feature_names, n_top_words)
topic_TopWords(svd, tfidf_feature_names, 20)


# In[9]:


n_samples = 2000
n_topics = 5
n_top_words = 20
n_features = 10000

def print_top_words(model, feature_names,n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()



# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   stop_words='english',ngram_range=(1, 2))
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                stop_words='english',ngram_range=(1, 2))
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)
topic_TopWords(nmf, tfidf_feature_names, 20)
print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
topic_TopWords(lda, tf_feature_names, 20)

svd = TruncatedSVD(n_components=n_topics)
t0 = time()
svd.fit(tfidf)  

print_top_words(svd, tfidf_feature_names, n_top_words)
topic_TopWords(svd, tfidf_feature_names, 20)

