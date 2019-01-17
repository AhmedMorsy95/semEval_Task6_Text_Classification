import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import Confusion_matrix

def run(train,test,train_labels,test_labels):
    clf = LinearSVC(random_state=0, C=0.6210526315789474,dual=False)
    clf.fit(train, train_labels)
    p = clf.predict(test)
    Confusion_matrix.print_matrix(test_labels, p)
    print("svm score : " , np.mean(p == test_labels))


def tune(train,train_labels):
    Cs = [float(x) for x in np.linspace(0.01,1 , num = 20)]
    # Cs = [0.001, 0.01, 0.1, 1, 10]
    param_grid = {'C': Cs}
    clf = LinearSVC()
    grid_search = GridSearchCV(clf, param_grid, cv=10,verbose=1)
    grid_search.fit(train,train_labels)
    print(grid_search.best_params_)



def plot(x_train,y_train,x_test,y_test):

        Cs = np.linspace(0.001, 5, 100, endpoint=True)
        train_results = []
        test_results = []
        for c in Cs:

           dt = LinearSVC(random_state=0, C=c)
           dt.fit(x_train, y_train)
           train_pred = dt.predict(x_train)
           false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
           roc_auc = auc(false_positive_rate, true_positive_rate)
           # Add auc score to previous train results
           train_results.append(roc_auc)
           y_pred = dt.predict(x_test)
           false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
           roc_auc = auc(false_positive_rate, true_positive_rate)
           # Add auc score to previous test results
           test_results.append(roc_auc)
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(Cs, train_results, 'b', label="Train")
        line2, = plt.plot(Cs, test_results, 'r', label="Test")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('Score')
        plt.xlabel('C')
        plt.show()
