import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint


import Confusion_matrix

def tune(train_features, train_labels):

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        pprint(random_grid)
        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        rf_random.fit(train_features, train_labels)
        print(rf_random.best_params_)


def run(train,test,train_labels,test_labels):

    rnd2 = RandomForestClassifier()
    rnd_clf = RandomForestClassifier(n_estimators=1400, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', max_depth=None, bootstrap=False)

    rnd_clf.fit(train,train_labels)
    p = rnd_clf.predict(test)
    Confusion_matrix.print_matrix(test_labels, p)


    rnd2.fit(train,train_labels)
    p2 = rnd2.predict(test)
    print("random forests score : " , np.mean(p == test_labels))

    print("random forests2 score : " , np.mean(p2 == test_labels))
