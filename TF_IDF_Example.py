'''
Author: Bhabajeet Kalita
Date: 12 - 09 - 2018
Description: Demonstration of TF-IDF (Term Frequency - Inverse Document Frequency)
'''
import nltk

dataset = {
    "tfidf-1.txt":open("tfidf-1.txt").read(),
    "tfidf-2.txt":open("tfidf-2.txt").read(),
    "tfidf-3.txt":open("tfidf-3.txt").read(),
    "tfidf-4.txt":open("tfidf-4.txt").read(),
    "tfidf-5.txt":open("tfidf-5.txt").read()
}
print(dataset.keys())


def tf(dataset,file_name):
    text = dataset[file_name]
    tokens = nltk.word_tokenize(text)
    fd = nltk.FreqDist(tokens)
    return fd


import math
def idf(dataset,term):
    count = [term in dataset[file_name] for file_name in dataset]
    inv_df = math.log(len(count)/sum(count))
    return inv_df

def tfidf(dataset,file_name,n):
    term_scores = {}
    file_fd = tf(dataset,file_name)
    for term in file_fd:
        if term.isalpha():
            idf_val = idf(dataset,term)
            tf_val = tf(dataset,file_name)[term]
            tfidf_val = tf_val*idf_val
            term_scores[term] = round(tfidf_val,2)
    return sorted(term_scores.items(),key=lambda x:x[1],reverse=True)[:n]

tfidf(dataset,"tfidf-1.txt",10)
for file_name in dataset:
    print("{0}: \n {1} \n".format(file_name,tfidf(dataset,file_name,5)))

import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt') # if necessary...


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


main = open("tfidf-1.txt")
main_file = main.read()

import glob
file_list = glob.glob("/Users/Gourhari/Desktop/TF_IDF/*")


for filename in file_list:
    print("Similarity between "+str(main.name)+" and "+str(filename)+" is : "+str(cosine_sim(main_file,open(filename).read())))

from nltk.corpus import conll2000, conll2002
print(conll2000.sents())
for tree in conll2000.chunked_sents()[:5]:
    print(tree)

# Install: pip install spacy && python -m spacy download en
import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en')

# Process whole documents
text = open('tfidf-1.txt').read()
doc = nlp(text)

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

# Determine semantic similarities
doc1 = nlp(u'the fries were grossly good')
doc2 = nlp(u'the fries were grossly bad')
doc1.similarity(doc2)
