from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import Confusion_matrix

def run(train,test,train_labels,test_labels):
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(train,train_labels)
    p = neigh.predict(test)
    Confusion_matrix.print_matrix(test_labels, predicted)
    print("K nearest neighbours clf score : " , np.mean(p == test_labels))
