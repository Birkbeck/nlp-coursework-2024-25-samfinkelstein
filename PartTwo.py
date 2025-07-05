import pandas as pd

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
