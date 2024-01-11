# target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
import time

import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import pandas as pd
import re

display=pd.options.display
display.max_columns=None
display.max_rows=None
display.width=None
display.max_colwidth=None

text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
def preprocess(text):
    text = re.sub(text_cleaning_re, " ", str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stopwords:
            tokens.append(token)

    return " ".join(tokens)

columns = ["target", "ids", "date", "flag", "user", "text"]
df = pd.read_csv("train_sentiment.csv",
                 encoding="ISO-8859-1",
                 names=columns)

# download stopwords
nltk.download("stopwords")
stopwords = set(stopwords.words("english"))
print(stopwords)

# clean text
t1 = time.time()
df.text = df.text.apply(lambda x: preprocess(x))
t2 = time.time()

print(f"text preprocessing time: {t2 - t1} seconds")

vocabularies = [chapter.split() for chapter in df.text]

w2v_model = Word2Vec(window=7,
                     min_count=10,
                     workers=8)
w2v_model.build_vocab(vocabularies)
words = list(w2v_model.wv.key_to_index.keys())
vocab_size = len(words)
print(f"Total vocabs: {vocab_size}")

t1 = time.time()
w2v_model.train(vocabularies, total_examples=len(vocabularies), epochs=32)
w2v_model.save("w2v_model")
t2 = time.time()
print(f"training time: {t2 - t1} seconds")