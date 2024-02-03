# dataset https://mahaljsp.ddns.net/wp-content/uploads/2024/01/DeadSimpleSpeechRecognizer.zip
import os

import keras.losses
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense
from keras.utils import to_categorical

from preprocess import save_data_to_array, get_train_test

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

max_pad_len = 11
nc = 3
# save_data_to_array(path="./data", max_pad_len=max_pad_len)

X_train, X_test, y_train, y_test = get_train_test()

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
print(X_train.shape)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, (2,2), activation="relu", input_shape=[20, max_pad_len, 1]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(nc, activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=["accuracy"],
              )

model.fit(X_train,
          y_train_hot,
          batch_size=100,
          epochs=2500,
          verbose=1,
          validation_data=(X_test, y_test_hot))
score = model.evaluate(X_test, y_test_hot)
print(score)
model.save("./ASR.h5")
