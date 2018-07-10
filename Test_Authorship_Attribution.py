
"""
Author: Bhabajeet Kalita

Designation: Masters Student, Applied Computer Science, Georg August University of Goettingen (2015-2018)

Title - Program to find the Authorship attribution of text

Description - ARI is a readability test value, designed to gauge the understandability of a text.

Date - 22/07/2017
"""
from collections import defaultdict
from preprocessing import read_corpus,tokenize
import os
from os.path import splitext
from os.path import basename
from math import log

#Function to read the file name
def read_file(filename):
    "Read the contents of FILENAME and return as a string."
    infile = open(filename) # windows users should use codecs.open after importing codecs
    contents = infile.read()
    infile.close()
    return contents

#Function to list the text files
def list_textfiles(directory):
    "Return a list of filenames ending in '.txt' in DIRECTORY."
    textfiles = []
    for filename in listdir(directory):
        if filename.endswith(".txt"):
            textfiles.append(directory + "/" + filename)
    return textfiles

scores = {"Hermans": 0.15, "Voskuil": 0.55, "Reve": 0.2, "Mulisch": 0.18, "Claus": 0.02}

#Function to find the maximum value of the scores
def classify(scores):
    return(max(scores, key=lambda k: scores[k]))

#Function to remove punctuation and transform every sentence into lowercase
def rem_punc_low(sentence):
    punctuation = '!@#$%^&*()_-+={}[]:;"\'|<>,.?/~`'
    makeitastring = ''.join(sentence).lower()
    for marker in punctuation:
        makeitastring = makeitastring.replace(marker, "")
    return makeitastring.split()

#Function to find the end of a sentence in the text
def end_of_sentence_marker(character):
    end = '.?!'
    if character in end:
        return True
    else:
        return False

#Function to split the sentences in the text
def extract_features(text):
    sentences = []
    start = 0
    for end, character in enumerate(text):
        if end_of_sentence_marker(character):
            sentence = text[start: end + 1]
            sentences.append(rem_punc_low(sentence))
            start = end + 1
            return sentences
        else:
            return rem_punc_low(text)

def predict_author(text, feature_database):
    "Predict who wrote this text."
    return classify(score(extract_features(text), feature_database))

def extract_author(filename):
    # insert your code here
    root, ext=os.path.splitext(filename)
    name= str(os.path.basename(root))
    return name[:name.index('-')]

#Function takes as argument author name and the words extracted using extract_features and adds these to our feature_database. The function should return a new updated version of the feature_database
def update_counts(author, text, feature_database):
    for feature in extract_features(text):
        feature_database[author][feature]+=1
    return feature_database

# do not modify the code below, for testing only!
feature_database = defaultdict(lambda: defaultdict(int))
feature_database = update_counts("Anonymous", "This was written with a lack of inspiration",
                                 feature_database)

#Function that extracts the author from the filename and adds the feature counts to the
def add_file_to_database(filename, feature_database):
    return update_counts(extract_author(filename),
                         extract_features(filename),
                         feature_database)

#Function that takes the name of a directory as input and add all files in this directory
def add_directory_to_database(directory, feature_database):
    # insert your code here
    file_names=list_textfiles(directory)
    for files in file_names:
        add_file_to_database(files,feature_database)
    return feature_database

def log_probability(feature_counts, features_sum, n_features):
    return log((feature_counts + 1.0) / (features_sum + n_features))

def score(features, feature_database):
    "Predict who wrote the document on the basis of the corpus."
    scores = defaultdict(float)
    #Code to calculate n_features
    n_features=len(set([item for sublist in [feature_database[author].keys() for author in feature_database] for item in sublist]))

    # compute the number of features in the feature database here
    for author in feature_database:
        feature_counts=0
        features_sum=0
        scores[author]=log_probability(feature_counts, features_sum, n_features)

        features_sum=0
        for feature in feature_database[author].keys():
            features_sum+=feature_database[author][feature]
        for feature in features:
            feature_counts = feature_database[author][feature]

    return scores

# do not modify the code below, for testing your answer only!
# It should return True if you did well!
features = ["the", "a", "the", "be", "book"]
feature_database = defaultdict(lambda: defaultdict(int))
feature_database["A"]["the"] = 2
feature_database["A"]["a"] = 5
feature_database["A"]["book"]= 1
feature_database["B"]["the"] = 5
feature_database["B"]["a"] = 1
feature_database["B"]["book"] = 6
print(abs(dict(score(features, feature_database))["A"] - -7.30734) < 0.001)
