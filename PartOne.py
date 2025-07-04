#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures
import spacy
from pathlib import Path
import pandas as pd
import os
import pickle
from collections import Counter
import pronouncing


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

def read_novels_list(path=Path.cwd() / "texts" / "novels"):
    noveldata = []
    filetype = '.txt'
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        purename = os.path.splitext(filename)[0]
        nameparts = purename.split('-')
        title = nameparts[0]
        author = nameparts[1]
        year = int(nameparts[2])
        with open(filepath, 'r', encoding = 'utf-8') as file:
            text = file.read()
        noveldata.append({'text': text, 'title': title, 'author': author, 'year': year})
    return noveldata

def read_novels(path=Path.cwd() / "texts" / "novels"):
    dataframe = pd.DataFrame(read_novels_list(path))
    dataframe = dataframe.sort_values('year').reset_index(drop=True)
    return dataframe

def nltk_ttr(text):
    tokens = RegexpTokenizer(r"\b\w+(?:'\w+)?\b").tokenize(text)
    tokencount = len(tokens)
    types = set(token.lower() for token in tokens)
    ttr = len(types)/len(tokens)
    return ttr


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""#? this extracts a dictionary from a df, no?
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def count_syl(word):
    lowercaseword = word.lower()
    phones = pronouncing.phones_for_word(lowercaseword)
    if phones:
        phonestring = phones[0]
        vowelcount = 0
        for character in phonestring:
            if character.isdigit():
                vowelcount += 1
        return max (1, vowelcount)
    else:
        vowels = 'aeiouy'
        syllablecount = 0
        previouslettervowel = False
        for letter in lowercaseword:
            if letter in vowels and not previouslettervowel:
                syllablecount += 1
                previouslettervowel = True
            elif letter not in vowels:
                previouslettervowel = False
        if lowercaseword.endswith('e') and syllablecount > 1 and lowercaseword[-2] not in vowels:
            syllablecount -= 1
        return max(1, syllablecount)
    
def fk_level(text):
    sentences = nltk.sent_tokenize(text)
    sentencecount = len(sentences)
    words = [word for word in nltk.word_tokenize(text) if any(character.isalpha() for character in word)]
    wordcount = len(words)
    syllablecount = 0
    for word in words:
        syllablecount += count_syl(word)
    fkscore = 0.39 * (wordcount / sentencecount) + 11.8 * (syllablecount / wordcount) - 15.59
    return round(fkscore, 2)
    
def flesh_kincaid(df):
    fkdict = {}
    listofnoveldata = df.values.tolist()
    for novel in listofnoveldata:
        fkdict[novel[1]] = fk_level(novel[0])
    return fkdict



def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    df['parsed'] = df['text'].apply(nlp)
    os.makedirs(store_path, exist_ok = True)
    path = store_path / out_name
    with open(path, 'wb') as file:
        pickle.dump(df, file)
    return df

def read_pickle(path=Path.cwd() / "pickles" /"name.pickle"):
    with open(path, 'rb') as file:
        df = pickle.load(file)
        return df

'''
def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results
    '''

def subjectfinder(verb):
    subjects = []
    for child in verb.children:
        if child.dep_ in ["nsubj", "nsubjpass"]:
            subjects.append(child)
    return subjects

def subjectcleaner(subject):
    normsubject = subject.text.strip().strip('-').lower()
    stopwordsset = set(stopwords.words("english"))
    if normsubject in stopwordsset:
        return None
    return normsubject

def subjects_by_verb_pmi(doc, targetverb):
    subjectverbpairs = []
    for sentence in doc.sents:
        for token in sentence:
            if token.lemma_.lower() == targetverb.lower() or token.text.lower() == targetverb.lower():
                subjects = subjectfinder(token)
                for subject in subjects:
                    cleansubject = subjectcleaner(subject)
                    if cleansubject and cleansubject != "_":
                        pair = (cleansubject,targetverb.lower())
                        subjectverbpairs.append(pair)
    finder = BigramCollocationFinder.from_documents(subjectverbpairs)
    toppairlist = finder.nbest(BigramAssocMeasures.pmi, 10)
    subjectlist = []
    for pair in toppairlist:
        subjectlist.append(pair[0])
    return subjectlist


def subjects_by_verb_count(doc, targetverb):
    subjectcounter = Counter()
    for sentence in doc.sents:
        for token in sentence:
            if token.lemma_.lower() == targetverb.lower() or token.text.lower() == targetverb.lower():
                subjects = subjectfinder(token)
                for subject in subjects:
                    cleansubject = subjectcleaner(subject)
                    if cleansubject and cleansubject != "_":
                        subjectcounter[cleansubject] += 1
    commonsubjects = []
    for subject, count in subjectcounter.most_common(10):
        commonsubjects.append(subject)
    return commonsubjects


def syntacticobjectcount(doc):
    objects = [token.dep_ for token in doc]
    objectcounter = Counter(objects).most_common(10)
    commonobjects = []
    for syntacticobject, count in objectcounter:
        commonobjects.append(syntacticobject)
    return commonobjects




if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.'''
    print(df.head())
    try:
        nltk.data.find('corpora/cmudict')
    except LookupError:
        nltk.download("cmudict")
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    parse(df)
    print(df.head())
    print(get_ttrs(df))
    print(flesh_kincaid(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    for i, row in df.iterrows():
        print(row["title"])
        print(syntacticobjectcount(row["parsed"]))
        print("\n")  
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")

