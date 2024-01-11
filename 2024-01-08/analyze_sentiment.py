import os
import pickle

import keras
from keras.utils import pad_sequences

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def get_label(score):
    label = "Neutral"
    if score <= 0.4:
        label = "Negative"
    elif score >= 0.7:
        label = "Positive"
    return label

model = keras.models.load_model("./sentiment_model")

with open("./eng_dictionary.pk1", "rb") as f:
    tokenizer = pickle.load(f)

while True:
    x = input("Input: ")
    if x == "quit": break
    x_test = pad_sequences(tokenizer.texts_to_sequences([x]), maxlen=300)
    score = model.predict([x_test])[0]
    label = get_label(score)
    print(score, label)