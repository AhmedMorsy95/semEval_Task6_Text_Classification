import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


def remove_noise(tweets):

    # noise related to twitter
    noise = ["url","user","@","#","-","!","?","rt","dm","retweet","rt","dm"]
    for i in range(0, len(tweets)):
        for j in range(0,len(noise)):
             tweets[i] = tweets[i].replace(noise[j] , "")

    #Shahenda's part
    porter_stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()

    wpt = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words('english')

    # abbreviations
    lookup_dict = {'rt': 'Retweet', 'dm': 'direct message', "awsm": "awesome", "luv": "love", "da": "the" , 'tbh': 'to be honest',"yup": "yes","wanna": "want",'lol': 'laugh out loud',}
    for i in range(0, len(tweets)):
        words = []
        words = wpt.tokenize(tweets[i])

        for word in words:
            if word.isalpha():
                if word.lower() in lookup_dict:
                    word = lookup_dict[word.lower()]

            #lemmatization
            word = wordnet_lemmatizer.lemmatize(word)

        tweets[i] = ' '.join(words)

    # remove stop words
    for i in range(0,len(tweets)):
        tweets[i] = tweets[i].lower()
        tweets[i] = (tweets[i].encode('ascii', 'ignore')).decode("utf-8") #remove emojis
        tokens = wpt.tokenize(tweets[i])
        filtered_tokens = [token for token in tokens if token not in stop_words]
        # re-create document from filtered tokens
        tweets[i] = ' '.join(filtered_tokens)


    return tweets
