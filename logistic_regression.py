from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report
import Confusion_matrix

def  run(train,test,train_labels,test_labels):
    log_reg = LogisticRegression()
    log_reg.fit(train, train_labels)
    p = log_reg.predict(test)
    target_names = ['OFF', 'NOT']
    #print(classification_report(test_labels, p, target_names=target_names))
    Confusion_matrix.print_matrix(test_labels, predicted)
    print("logistic regression score : " , np.mean(p == test_labels))
