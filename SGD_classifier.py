from sklearn.linear_model import SGDClassifier
import numpy as np


def run(train,test,train_labels,test_labels):
    sgd_clf = SGDClassifier(random_state=42,max_iter=1000)
    sgd_clf.fit(train, train_labels)
    p = sgd_clf.predict(test)
    print("Stochastic gradient Descent Score : " ,np.mean(p == test_labels))
