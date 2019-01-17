from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report
import Confusion_matrix
from sklearn.model_selection import GridSearchCV


def  run(train,test,train_labels,test_labels):
    log_reg = LogisticRegression(C=1.1894736842105262,penalty='l1')
    log_reg.fit(train, train_labels)
    p = log_reg.predict(test)
    target_names = ['OFF', 'NOT']
    #print(classification_report(test_labels, p, target_names=target_names))
    Confusion_matrix.print_matrix(test_labels, p)
    print("logistic regression score : " , np.mean(p == test_labels))

def tune(train,labels):
        param_grid = {'C': [float(x) for x in np.linspace(1.1, 1.2, num = 20)], 'penalty': ['l1', 'l2']}
        clf = LogisticRegression()
        gridsearch = GridSearchCV(clf, param_grid,verbose=1,n_jobs=-1)
        gridsearch.fit(train,labels)
        print(gridsearch.best_params_)
