from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import math
import random
# reads data and divides it into training and testing data

def get_data(ratio):
    # arrays we will return
    tweets = []
    labels = []
    tests = []
    test_labels = []

    with open('offenseval-training-v1.tsv') as f:
        lines = f.read().split('\n')[:-1]
        for i, line in enumerate(lines):
            if i == 0: # header
                column_names = line.split("\t")
            else:
                # only interested in non NULL data belonging to sub-task A
                cur_line = line.split("\t")
                if cur_line[2] != "NULL":
                    tweets.append(cur_line[1])
                    if(cur_line[2] == "OFF"):
                        labels.append(1)
                    else:
                        labels.append(0)


    # split data ratio is the % of training data , the rest is for testing isa
    #print(len(tweets))
    n = math.ceil(ratio * len(tweets))
    tests = tweets[n:]
    test_labels = labels[n:]
    tweets = tweets[0:n]
    labels = labels[0:n]
    #print(len(tweets) , len(tests))

    return tweets,labels,tests,test_labels
