from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import Confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def run(train,test,train_labels,test_labels):
    neigh = KNeighborsClassifier(n_neighbors=17)
    neigh.fit(train,train_labels)
    p = neigh.predict(test)
    #Confusion_matrix.print_matrix(test_labels, p)
    print("K nearest neighbours clf score : " , np.mean(p == test_labels))


# def plot(x_train,y_train,x_test,y_test):
#
#         max_depths =  [int(x) for x in np.linspace(1, 20, num = 20)]
#         print(max_depths)
#         train_results = []
#         test_results = []
#         for max_depth in max_depths:
#            dt = KNeighborsClassifier(n_neighbors=max_depth)
#            dt.fit(x_train, y_train)
#            train_pred = dt.predict(x_train)
#            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
#            roc_auc = auc(false_positive_rate, true_positive_rate)
#            # Add auc score to previous train results
#            train_results.append(roc_auc)
#            y_pred = dt.predict(x_test)
#            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
#            roc_auc = auc(false_positive_rate, true_positive_rate)
#            # Add auc score to previous test results
#            test_results.append(roc_auc)
#         from matplotlib.legend_handler import HandlerLine2D
#         line1, = plt.plot(max_depths, train_results, 'b', label="Train")
#         line2, = plt.plot(max_depths, test_results, 'r', label="Test")
#         plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
#         plt.ylabel('Score')
#         plt.xlabel('K')
#         plt.show()
