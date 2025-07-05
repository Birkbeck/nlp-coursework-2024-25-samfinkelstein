import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import train_test_split

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
df2 = speakerremover(df)
df3 = fourpartysystem(df)
df4 = speechonly(df)
df5 = noshortspeeches(df)
print(dimensions(df5))

'''
Vectorise the speeches using TfidfVectorizer from scikit-learn. Use the default
parameters, except for omitting English stopwords and setting max_features to
3000. Split the data into a train and test set, using stratified sampling, with a
random seed of 26.
'''

def vectorize(df):
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 3000)
    vectors = vectorizer.fit_transform(df['speech'])
    return vectors