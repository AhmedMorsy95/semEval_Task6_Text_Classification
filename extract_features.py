from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from gensim.models import Word2Vec
import gensim
import numpy as np

def get_features_TF_IDF(train,test):
    vectorizer = TfidfVectorizer(min_df=0., max_df=1., use_idf=True,  ngram_range=(1, 1))
    X = vectorizer.fit_transform(train)
    X_test = vectorizer.transform(test)
    return X,X_test

def word2vec(train,test):
    s = []
    for i in range(0,len(train)):
        x = train[i].split(' ')
        s.append(x)


    t = []
    for i in range(0,len(test)):
        x = test[i].split(' ')
        t.append(x)

    model = gensim.models.Word2Vec(s, min_count=1)
    model2 = gensim.models.Word2Vec(t, min_count=1)

    train_features = []
    test_features = []
    for i in s:
        lst1 = []
        for j in i:
            sum = 0
            word_vector = model[j]
            for k in word_vector:
                sum = sum + k
            lst1.append(sum)
        train_features.append(lst1)

    for i in t:
        lst1 = []
        for j in i:
            sum = 0
            word_vector = model2[j]
            for k in word_vector:
                sum = sum + k
            lst1.append(sum)
        test_features.append(lst1)



    x = np.array(train_features)
    y = np.array(test_features)
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    print(y.shape)

    return x,y
