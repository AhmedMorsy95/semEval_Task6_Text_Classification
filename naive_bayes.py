from sklearn.naive_bayes import MultinomialNB
import numpy as np
import Confusion_matrix


def run_naive_bayes(train,test,train_labels,test_labels):
    # train labeled data
    clf = MultinomialNB().fit(train, train_labels)

    # predict test data
    predicted = clf.predict(test)
    Confusion_matrix.print_matrix(test_labels, predicted)
    # print score
    print("naive bayes score : " , np.mean(predicted == test_labels))
