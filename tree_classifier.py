from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint # pretty print
import Confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def tune(train_features, train_labels):


        max_depth = [int(x) for x in np.linspace(2, 30, num = 15)]
        min_samples_splits = np.linspace(0.0001, 1.0, 200)
        min_samples_leafs = np.linspace(0.0001, 0.5, 200)
        max_features = list(range(1,train_features.shape[1]))

        # Create the random grid
        random_grid = {
                       'min_samples_split' : min_samples_splits,
                       'max_depth': max_depth,
                       'min_samples_leaf' : min_samples_leafs,
                       'max_features' : max_features
                       }

        pprint(random_grid)
        rf = DecisionTreeClassifier()
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 1000, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        #  njobs : number of jobs to run in paraller , -1 means use all processors
        #  verbose  : You can think of it as asking the program to "tell me everything about what you are doing all the time". Controls the verbosity: the higher, the more messages.
        # random_state : Pseudo random number generator state used for random uniform sampling from lists of possible values


        rf_random.fit(train_features, train_labels)
        print(rf_random.best_params_)


def run(train,test,train_labels,test_labels):
    tree_clf = DecisionTreeClassifier(max_depth=20,min_samples_split=0.10190909090909092,min_samples_leaf=0.001,max_features=4260)
    tree_clf.fit(train, train_labels)
    p = tree_clf.predict(test)
    Confusion_matrix.print_matrix(test_labels, p)
    print("Tree classifier score : " , np.mean(p == test_labels))


def plot(x_train,y_train,x_test,y_test):

        max_depths = np.linspace(1, 100, 50, endpoint=True)
        train_results = []
        test_results = []
        for max_depth in max_depths:
           dt = DecisionTreeClassifier(max_depth=max_depth)
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
        line1, = plt.plot(max_depths, train_results, 'b', label="Train")
        line2, = plt.plot(max_depths, test_results, 'r', label="Test")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('Score')
        plt.xlabel('Tree depth')
        plt.show()
