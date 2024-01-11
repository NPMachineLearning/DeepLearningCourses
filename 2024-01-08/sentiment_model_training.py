import os.path
import pickle
import shutil
from time import time

import nltk
import numpy as np
from gensim.models import Word2Vec
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Embedding, Dropout, LSTM, Dense
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import pandas as pd
import re
from keras.utils import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

columns = ["target", "ids", "date", "flag", "user", "text"]
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text):
    text = re.sub(text_cleaning_re, " ", str(text).lower()).strip()
    tokens = []
    for token in tokens:
        if token not in stopwords:
            tokens.append(token)
    return " ".join(tokens)

def decode_sentiment(label):
    return decode_map[int(label)]

nltk.download("stopwords")
stopwords = stopwords.words("english")
print("Reading csv ....")
df = pd.read_csv("./train_sentiment.csv",
                 encoding="ISO-8859-1",
                 names=columns)
df.text = df.text.apply(lambda x: preprocess(x))
decode_map = {0:"Negative", 2:"Neutral", 4:"Positive"}
df.target = df.target.apply(lambda x: decode_sentiment(x))

with open("./eng_dict.pkl", "rb") as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer.word_index)
print(vocab_size)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
weights = np.zeros([vocab_size, 100])
w2v_model = Word2Vec.load("./w2v_model")
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        weights[i] = w2v_model.wv[word]

X_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=300)
X_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=300)
encoder = LabelEncoder()
encoder.fit(df_train.target.tolist())
y_train = encoder.transform(df_train.target.tolist())
y_test = encoder.transform(df_test.target.tolist())
print(y_train)
y_train = y_train.reshape([-1, 1])
y_test = y_test.reshape([-1, 1])

model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                    output_dim=100,
                    weights=[weights],
                    input_length=300,
                    trainable=False))
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="Adam",
              metrics=["accuracy"])
callback = [ReduceLROnPlateau(monitor="val_loss", patience=5, cooldown=0),
            EarlyStopping(monitor="val_acc", min_delta=1e-4, patience=5)]

BATCH_SIZE = 32

history = model.fit(X_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=8,
                    validation_split=0.1,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks=callback)

path = "./sentiment_analyzer_model"
if os.path.exists(path):
    shutil.rmtree(path)
model.save(path)
score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)

plt.figure()
epochs = len(history.history["loss"])
plt.plot(epochs, history.history["loss"], label="Train loss")
plt.plot(epochs, history.history["val_loss"], label="Val loss")
plt.plot(epochs, history.history["accuracy"], label="Train accuracy")
plt.plot(epochs, history.history["val_accuracy"], label="Val accuracy")
plt.legend()
plt.show()