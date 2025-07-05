import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

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

def vectorize(trainX, testX):
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 3000)
    trainXvectors = vectorizer.fit_transform(trainX)
    testXvectors = vectorizer.fit_transform(testX)
    return trainXvectors, testXvectors

def randomforest(trainX, trainY, testX, testY):
    classifier = RandomForestClassifier(n_estimators = 300, random_state = 26)
    classifier.fit(trainX, trainY)
    predictY = classifier.predict(testX)

    #return predictions and classifier name

def classifierresults(testY, predY, classifiername):
    report = classification_report(testY, predY)
    f1score = print(f'Macro-average F1 Score for {classifiername}: {f1_score(testY, predY, average = 'macro')}')
    printreport = print(f'Classification Report for {classifiername}: {report}')
    return f1score, printreport
    




