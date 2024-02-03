import os

import numpy as np
from keras import models

from preprocess import wav2mfcc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
max_pad_len = 11
path = "./data/happy"
model = models.load_model("ASR_adam.h5")
for file in os.listdir(path):
    file_path  = os.path.join(path, file)
    mfcc = wav2mfcc(file_path, max_pad_len)
    mfcc = mfcc.reshape(1, 20, max_pad_len, 1)
    pred = model.predict(mfcc)
    print(f"Prediction: {np.argmax(pred)}")
