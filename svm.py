import numpy as np
from sklearn.svm import LinearSVC

import Confusion_matrix

def run(train,test,train_labels,test_labels):
    clf = LinearSVC(random_state=0, tol=1e-9)
    clf.fit(train, train_labels)
    p = clf.predict(test)
    Confusion_matrix.print_matrix(test_labels, predicted)

    print("svm score : " , np.mean(p == test_labels))
