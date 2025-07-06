import pandas as pd
#pd.set_option('display.max_columns', None)   -- used for custom tokenizer stopword decisions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn import svm
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def readhansard():
    df = pd.read_csv("p2-texts/hansard40000.csv")
    return df

def labour(df):
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
    return df

def speakerremover(df):
    df = df[df['party'] != 'Speaker']
    return df

def fourpartysystem(df):
    top4parties = df['party'].value_counts().head(4).index
    df = df[df['party'].isin(top4parties)]
    return df

def speechonly(df):
    df = df[df['speech_class'] == 'Speech']
    return df

def noshortspeeches(df):
    df = df[df['speech'].str.len() >= 1000]
    return df

def dimensions(df):
    return df.shape

df = readhansard()
df1 = labour(df)
df2 = speakerremover(df1)
df3 = fourpartysystem(df2)
df4 = speechonly(df3)
df5 = noshortspeeches(df4)
print(dimensions(df5))

'''
Vectorise the speeches using TfidfVectorizer from scikit-learn. Use the default
parameters, except for omitting English stopwords and setting max_features to
3000. Split the data into a train and test set, using stratified sampling, with a
random seed of 26.
'''
def traintestsplitter(df):
    speechtrain, speechtest, partytrain, partytest = train_test_split(df['speech'], df['party'], random_state = 26, stratify = df['party'])
    return speechtrain, speechtest, partytrain, partytest
'''
def vectorize1(trainX, testX):
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 3000)
    trainXvectors = vectorizer.fit_transform(trainX)
    testXvectors = vectorizer.transform(testX)
    return trainXvectors, testXvectors

def randomforest(trainX, trainY, testX):
    classifier = RandomForestClassifier(n_estimators = 300, random_state = 26)
    classifier.fit(trainX, trainY)
    predictY = classifier.predict(testX)
    return predictY

def svmC(trainX, trainY, testX):
    classifier = svm.SVC(kernel = 'linear')
    classifier.fit(trainX, trainY)
    predictY = classifier.predict(testX)
    return predictY

def classifierresults(testY, predY, classifiername):
    print(f'Macro-average F1 Score for {classifiername}:')
    print(f1_score(testY, predY, average = 'macro'))
    print(f'Classification Report for {classifiername}')
    print(classification_report(testY, predY))
    
def resulttester(df, vectorizerfunction):    
    trainX, testX, trainY, testY = traintestsplitter(df)
    trainXvectors, testXvectors = vectorizerfunction(trainX, testX)
    rfpreds = randomforest(trainXvectors, trainY, testXvectors)
    svmpreds = svmC(trainXvectors, trainY, testXvectors)
    classifierresults(testY, rfpreds, 'Random Forest Classifier')
    classifierresults(testY, svmpreds, 'SVM')

resulttester(df5, vectorize1)

def vectorize2(trainX, testX):
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 3000, ngram_range = (1, 3))
    trainXvectors = vectorizer.fit_transform(trainX)
    testXvectors = vectorizer.transform(testX)
    return trainXvectors, testXvectors

resultreporter(df5, vectorize2)
'''

def customtokenizer(text):
    doc = nlp(text)
    importanttokens = []
    customstopwords = ['support', 'bill', 'time', 'say', 'know', 'want', 'come', 'place', 'good', 'today', 'amendment', 'debate', 'way', 'ensure', 'issue', 'public', 'think', 'take', 'include', 'change']
    for token in doc:
        if not token.is_stop and len(token) > 2 and token.is_alpha and token.lemma_.lower() not in customstopwords:
            importanttokens.append(token.lemma_.lower())
    return importanttokens

def toptokenfinder(tokenlist):
    top30 = Counter(tokenlist).most_common(30)
    top30tokens = []
    for pair in top30:
        top30tokens.append(pair[0])
    return top30tokens

def stopwordchecker(tokenlist, df):
    partiestopwords = pd.DataFrame(index = df.index)
    for token in tokenlist:
        partiestopwords[token] = df['speech'].str.lower().str.count(token)
    partiestopwords['party'] = df['party']
    totalwordseach = df.groupby('party')['speech'].apply(lambda x: x.str.split().str.len().sum())
    stopwordreport = partiestopwords.groupby('party').sum()
    normalisedstopwordreport = stopwordreport.div(totalwordseach, axis = 0) * 1000
    return normalisedstopwordreport

def vectorize3(trainX, testX):
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 3000, ngram_range = (1, 3), tokenizer = customtokenizer)
    trainXvectors = vectorizer.fit_transform(trainX)
    testXvectors = vectorizer.transform(testX)
    return trainXvectors, testXvectors
#code below was used to identify suitable stopwords for my custom tokenizer
#allspeechtokens = df5['speech'].apply(customtokenizer).sum()
#toptokens = toptokenfinder(allspeechtokens)
#stopwordcheck = stopwordchecker(toptokens, df5)
#print(stopwordcheck)