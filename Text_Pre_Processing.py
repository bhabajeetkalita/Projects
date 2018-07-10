'''
Author: Bhabajeet Kalita
Date: 12 - 06 - 2018
Description: Demonstration of Text Pre-Processing Steps
'''

import nltk
text = "She looked at   her father's arm-chair."
text_fr = "Qu'est-ce que c'est?"
text.split(' ')
text_fr.split(' ')
from sklearn.feature_extraction.text import CountVectorizer
CountVectorizer().build_tokenizer()(text)
from nltk.tokenize import word_tokenize
word_tokenize(text)
#from nltk.tokenize.punkt import PunktWordTokenizer
#tokenizer = PunktWordTokenizer()
#tokenizer.tokenize(text)

#Stemming
from nltk.stem.snowball import GermanStemmer
stemmer=GermanStemmer()
words=["Wald", "Walde", "Wälder", "Wäldern", "Waldes","Walds"]
stemmer.stem("Waldi")
#[stemmer.stem(w) for w in words]

#Chunking
import os
import numpy as np
corpus_path = os.path.join('/Users/Gourhari/Documents/Py/data', 'french-tragedy')
sorted(os.listdir(corpus_path))[0:5]
tragedy_filenames = [os.path.join(corpus_path, fn) for fn in sorted(os.listdir(corpus_path))]

def split_text(filename, n_words):
    """Split a text into chunks approximately `n_words` words in length."""
    input = open(filename, 'r')
    words = input.read().split(' ')
    input.close()
    chunks = []
    current_chunk_words = []
    current_chunk_word_count = 0
    for word in words:
        current_chunk_words.append(word)
        current_chunk_word_count += 1
        if current_chunk_word_count == n_words:
            chunks.append(' '.join(current_chunk_words))
            current_chunk_words = []
            current_chunk_word_count = 0
    chunks.append(' '.join(current_chunk_words))
    return chunks
tragedy_filenames = [os.path.join(corpus_path, fn) for fn in sorted(os.listdir(corpus_path))]

tragedy_filenames.sort()

chunk_length = 1000

chunks = []

for filename in tragedy_filenames:
    chunk_counter = 0
    texts = split_text(filename, chunk_length)
    for text in texts:
        chunk = {'text': text, 'number': chunk_counter, 'filename': filename}
        chunks.append(chunk)
        chunk_counter += 1
#print(len(chunks))

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=5, max_df=.95)
dtm = vectorizer.fit_transform([c['text'] for c in chunks])
vocab = np.array(vectorizer.get_feature_names())
#print(dtm)

output_dir = '/Users/Gourhari/Documents/Py/data/FT_OP/'

for chunk in chunks:
    basename = os.path.basename(chunk['filename'])
    fn = os.path.join(output_dir,
    "{}{:04d}".format(basename, chunk['number']))
    with open(fn, 'w') as f:
        f.write(chunk['text'])

authors = [os.path.basename(filename).split('_')[0] for filename in tragedy_filenames]
authors = np.array(authors)
first_author = sorted(set(authors))[0]
vectorizer = CountVectorizer(input='filename')
dtm = vectorizer.fit_transform(tragedy_filenames).toarray()
vocab = np.array(vectorizer.get_feature_names())
authors = np.array([os.path.basename(filename).split('_')[0] for filename in tragedy_filenames])
authors_unique = sorted(set(authors))
#print(authors_unique)
#print(vocab)
#dtm_authors = np.zeros((len(authors_unique), len(vocab)))
#print(dtm_authors)

for i, author in enumerate(authors_unique):
    dtm_authors[i, :] = np.sum(dtm[authors==author, :], axis=0)
#print(dtm_authors[1:])

import pandas
authors=[os.path.basename(filename).split('_')[0] for filename in tragedy_filenames]
dtm_authors=pandas.DataFrame(dtm).groupby(authors).sum().values
#print(dtm_authors)

import itertools
import operator
texts=[]
for filename in tragedy_filenames:
    author=os.path.basename(filename).split('_')[0]
    texts.append(dict(filename=filename,author=author))
#print(texts)
texts = sorted(texts, key=operator.itemgetter('author'))

grouped_data = {}
for author, grouped in itertools.groupby(texts, key=operator.itemgetter('author')):
    grouped_data[author] = ','.join(os.path.basename(t['filename']) for t in grouped)

#grouped_data
texts=[]
for i, filename in enumerate(tragedy_filenames):
    author = os.path.basename(filename).split('_')[0]
    termfreq = dtm[i, :]
    print(termfreq)
    texts.append(dict(filename=filename, author=author, termfreq=termfreq))

texts = sorted(texts, key=operator.itemgetter('author'))
termfreqs = []
for author, group in itertools.groupby(texts, key=operator.itemgetter('author')):
    termfreqs.append(np.sum(np.array([t['termfreq'] for t in group]), axis=0))

dtm_authors = np.array(termfreqs)
np.testing.assert_array_almost_equal(dtm_authors_method_groupby, dtm_authors_method_numpy)

import matplotlib

import matplotlib.pyplot as plt

from sklearn.manifold import MDS

from sklearn.metrics.pairwise import cosine_similarity

dist = 1 - cosine_similarity(dtm_authors)

mds = MDS(n_components=2, dissimilarity="precomputed")

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

 xs, ys = pos[:, 0], pos[:, 1]

names = sorted(set(authors))

for x, y, name in zip(xs, ys, names):
    color = matplotlib.cm.summer(names.index(name))
    plt.scatter(x, y, c=color)
    plt.text(x, y, name)

plt.show()
