import os
import shutil

import keras.optimizers
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential, Input, Model
from keras.datasets import mnist
from keras.layers import Dense, LeakyReLU, BatchNormalization, Flatten, Reshape

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        optimizer = keras.optimizers.Adam(0.0002, 0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="binary_crossentropy",
                                   optimizer=optimizer,
                                   metrics=["accuracy"])
        self.generator = self.build_generator()
        input = Input(shape=self.latent_dim)
        img = self.generator(input)

        self.discriminator.trainable = False
        output = self.discriminator(img)

        self.combine = Model(input, output)
        self.combine.compile(loss="binary_crossentropy",
                             optimizer=optimizer)
    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation="tanh"))
        model.add(Reshape(self.img_shape))
        input = Input(shape=self.latent_dim)
        output = model(input)
        return Model(input, output)
    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        input = Input(shape=self.img_shape)
        output = model(input)
        return Model(input, output)
    def train(self, epochs, batch_size=128, sample_interval=50):
        (x_train, _), (_, _) = mnist.load_data()
        x_train = x_train / 127.5 - 1.
        x_train = np.expand_dims(x_train, axis=3)
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_imgs = x_train[idx]
            noise = np.random.normal(0, 1, size=(batch_size, self.latent_dim))
            fake_imgs = self.generator(noise)

            d_loss_real = self.discriminator.train_on_batch(real_imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.combine.train_on_batch(noise, valid)
            print(f"Epoch:{epoch} | Loss:{d_loss[0]} | Accuracy:{100*d_loss[1]}% | g_loss:{g_loss}")
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

        self.generator.save("mnist_gan")
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, size=(r*c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = gen_imgs * 0.5 + 0.5
        fig, axs = plt.subplots(r, c)
        idx = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[idx, :,:,0], cmap="gray")
                axs[i, j].axis(False)
                idx += 1
        fig.savefig(f"./images/img_{epoch}.jpg")
        plt.close()

if __name__ == "__main__":
    path = "./images"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    gan = GAN()
    gan.train(epochs=20000, batch_size=128, sample_interval=100)