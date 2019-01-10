import random
import math
import Data_cleaner
import extract_features
import naive_bayes
import logistic_regression
import warnings
import tree_classifier
import svm
import random_forest
import Kneighbours_Clf

# divides data into k chunks and prompts score

def validate(k,data,labels):
    list = []
    for i in range(0,len(data)):
        list.append((data[i],labels[i]))

    random.shuffle(list)

    print(k , " fold Cross Validation\n\n")

    chunk_size = math.floor(len(list)/k)
    #print(chunk_size)

    for i in range(0,k):
        test = []
        train = []
        # divided our data k-1 for training  , 1 for testing
        for j in range(0,len(list)):
            if math.floor(j/chunk_size) == i:
                test.append(list[j])
            else:
                train.append(list[j])

        #print(len(test),len(train))
        if i == 0:
            print("1st test")
        elif i == 1:
            print("2nd test")
        elif i == 2:
            print("3rd test")
        else:
            print(i+1,"th test")

        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        # get divided and cleaned data
        train_data,train_labels,test_data,test_labels = preprocess(test, train)

        #get features
        train_features,test_features = extract_features.get_features_TF_IDF(train_data,test_data)

        # runs classifier
        run_clf(train_features,train_labels,test_features,test_labels)


def preprocess(test,train):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # separate data and labels

    for i in range(0,len(train)):
        train_data.append(train[i][0])
        train_labels.append(train[i][1])


    for i in range(0,len(test)):
        test_data.append(test[i][0])
        test_labels.append(test[i][1])

    # clean data
    test_data = Data_cleaner.remove_noise(test_data)
    train_data = Data_cleaner.remove_noise(train_data)

    return train_data,train_labels,test_data,test_labels


def run_clf(train_features,train_labels,test_features,test_labels):
        # naive_bayes.run_naive_bayes(train_features, test_features, train_labels, test_labels)
        # logistic_regression.run(train_features, test_features, train_labels, test_labels)
        tree_classifier.run(train_features, test_features, train_labels, test_labels)
        # svm.run(train_features, test_features, train_labels, test_labels)
        # random_forest.run(train_features, test_features, train_labels, test_labels)
        print("\n")
