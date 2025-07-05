import pandas as pd
import os

def read_hansard():
    df = pd.read_csv("p2-texts/hansard40000.csv")
    return df

