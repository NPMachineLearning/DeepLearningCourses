import pickle
from time import time

import nltk
from keras.utils import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import pandas as pd
import re

texts = ["I am Nelson", "This is test", "This is bad"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
print(tokenizer.word_index)

s = ["This is garden"]
print(tokenizer.texts_to_sequences(s))

# pad sequence into same length
print(pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=10))
print(pad_sequences(tokenizer.texts_to_sequences(s), maxlen=10))

columns = ["target", "ids", "date", "flag", "user", "text"]
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text):
    text = re.sub(text_cleaning_re, " ", str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stopwords:
            tokens.append(token)
    return " ".join(tokens)

nltk.download("stopwords")
stopwords = stopwords.words("english")
print("Reading csv ....")
df = pd.read_csv("./train_sentiment.csv",
                 encoding="ISO-8859-1",
                 names=columns)
df.text = df.text.apply(lambda  x: preprocess(x))

print("Tokenizing.....")
t1 = time()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)

t2 = time()
print(f"Tokenization time: {t2 - t1} seconds")
print(f"vocab size: {len(tokenizer.word_index)}")
pickle.dump(tokenizer, open("eng_dict.pkl", "wb"), protocol=0)