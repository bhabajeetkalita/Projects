'''
Author: Bhabajeet Kalita
Date: 19/11/2017
Description: Master Thesis Backend Entity Disambiguation
'''
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# In[1]:


import re
import os
import sys
import time
import glob
import nltk
import string
import codecs
import wikipedia
from string import*
from os import listdir
import multiprocessing
from itertools import groupby
from nltk.corpus import stopwords
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
#nltk.download('punkt') # if necessary...
# lxml parser

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


st = StanfordNERTagger('./static/other_files/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz','./static/other_files/stanford-ner-2014-06-16/stanford-ner.jar',encoding='utf-8')

def disambiguate(text_document_name,main_directory,download_directory,tag_search):
    # tag_search is either PERSON or LOCATION
    if tag_search not in ('PERSON','LOCATION'):
        raise ValueError("tag_search parameter can only have values 'PERSON' or 'LOCATION' ")
    tagged_persons=[]
    with codecs.open(os.path.join(main_directory,text_document_name),"r") as text_document:
        text = text_document.read()
        tokenized = nltk.word_tokenize(text)
        filtered_words = [word for word in tokenized if word not in stopwords.words('english')]
        normalized_words = [word for word in filtered_words if word.isalpha()]
        tokens = st.tag(normalized_words)
        for tag in tokens:
            if tag[1]==tag_search: tagged_persons.append(tag[0])
        tagged_persons = list(set(tagged_persons))
        for tagged_person in tagged_persons:
            wiki_persons = wikipedia.search(tagged_person)
            d = {}
            for wiki_person in wiki_persons:
                try:
                    d[wiki_person] = cosine_sim(text,wikipedia.summary(wiki_person))
                except wikipedia.exceptions.PageError as f:
                    d[wiki_person] = 0
                except wikipedia.exceptions.DisambiguationError as e:
                    for wiki_person_option in e.options:
                        try:
                            d[wiki_person_option] = cosine_sim(text,wikipedia.summary(wiki_person_option))
                        except wikipedia.exceptions.DisambiguationError as b:
                            pass
                        except wikipedia.exceptions.PageError as c:
                            d[wiki_person_option] = 0
            url = wikipedia.page(max(d.keys(), key=lambda k: d[k])).url
            text = text.replace(tagged_person,"<{} url =  ".format(tag_search)+url+" > "+tagged_person+" </{}>".format(tag_search))
        with codecs.open(os.path.join(download_directory,"{}_{}".format(tag_search,text_document_name)),"w") as annotated_file:
            annotated_file.write(text)
