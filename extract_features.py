from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

def get_features_TF_IDF(train,test):
    vectorizer = TfidfVectorizer(min_df=0., max_df=1., use_idf=True,  ngram_range=(1, 1))
    X = vectorizer.fit_transform(train)
    X_test = vectorizer.transform(test)
    return X,X_test
