from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import Data_cleaner
import read_data
import gensim
import pandas
import extract_features
import naive_bayes
import SGD_classifier
import logistic_regression
import warnings
import tree_classifier
import svm
import random_forest
import Kneighbours_Clf
import k_fold_Cross_validation
from sklearn.neural_network import MLPClassifier


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    #read data
    tweets, labels, tests,test_labels = read_data.get_data(0.8)

    all_data = tweets + tests
    all_labels = labels + test_labels
    #k_fold_Cross_validation.validate(5, all_data, all_labels)

    trainDF = pandas.DataFrame()
    testDF = pandas.DataFrame()
    #
    # #remove noise
    # #pandas dataframe is a 2D array which can have several column_names and supports mathematical operations on rows and columns
    trainDF['text'] = Data_cleaner.remove_noise(tweets)
    trainDF['labels'] = labels
    testDF['tests'] = Data_cleaner.remove_noise(tests)
    testDF['test_labels'] = test_labels
    #print(trainDF)
    # # extract features from text and test
    train_features,test_features = extract_features.get_features_TF_IDF(trainDF['text'],testDF['tests'])

    # multi layer perceptron
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    # clf.fit(train_features,trainDF['labels'])
    # p = clf.predict(test_features)

    # print(np.mean(p == testDF['test_labels']))

    # run our first classifier naive naive_bayes
    #naive_bayes.run_naive_bayes(train_features, test_features,trainDF['labels'], testDF['test_labels'])
    #
    #
    # SGD_classifier
    #SGD_classifier.run(train_features, test_features,trainDF['labels'], testDF['test_labels'])
    #
    # #
    # # logistic regression
    #logistic_regression.run(train_features, test_features,trainDF['labels'], testDF['test_labels'])
    #
    #
    # # tree classifier
    #tree_classifier.tune(train_features,trainDF['labels'])
    #tree_classifier.plot(train_features,trainDF['labels'],test_features,testDF['test_labels'])
    tree_classifier.run(train_features,test_features,trainDF['labels'],testDF['test_labels'])
    #
    # # supported vector machines linear classifier ,tolerance
    # svm.run(train_features, test_features,trainDF['labels'], testDF['test_labels'])
    #
    #
    # # random forests
    #random_forest.tune(train_features, trainDF['labels'])
    #random_forest.run(train_features, test_features,trainDF['labels'], testDF['test_labels'])
    #
    #
    # # k nearest neighbour clf
    # Kneighbours_Clf.run(train_features, test_features,trainDF['labels'], testDF['test_labels'])


if __name__== "__main__":
    main()
