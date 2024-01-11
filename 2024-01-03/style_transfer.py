import os
import shutil

import cv2
import keras.optimizers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.optimizers import SGD

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def deprocess_image(img):
    # alter image brightness
    img = img.reshape((height, width, 3))[:,:,::-1].copy()
    # img[:,:,0] += 123.76
    # img[:,:,1] += 116.779
    # img[:,:,2] += 103.939
    img[:, :, 0] += 123
    img[:, :, 1] += 116
    img[:, :, 2] += 103
    return np.clip(img, 0, 255).astype(np.uint8)
def gram_matrix(x):
    # calcualte gram matrix
    x = tf.reshape(x, (x.shape[1], x.shape[2], x.shape[3]))
    x = tf.transpose(x, (2,0,1))
    features = tf.reshape(x, (x.shape[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style_feature, combination_feature):
    # total loss of style transfer
    s = gram_matrix(style_feature)
    c = gram_matrix(combination_feature)
    channels = 3
    size = width * height
    # (4 * (channels ** 2) * (size ** 2))是官網建議的，可以自已調整看看，可以看成是一個常數
    return tf.reduce_sum(tf.square(s - c)) / (4 * (channels ** 2) * (size ** 2))

def compute_loss_and_grads(combination_image, base_image, style_image):
    # loss and grads
    with tf.GradientTape() as tape:
        base_features = model(base_image)
        base_feature = base_features[content_layer_name]
        loss = tf.zeros(shape=())
        combination_features = model(combination_image)
        combination_feature = combination_features[content_layer_name]
        # calculate block5_conv2 loss
        loss = loss + tf.reduce_sum(tf.square(combination_feature - base_feature)) * content_weight

        style_features = model(style_image)
        for layer in style_layer_names:
            style_feature = style_features[layer]
            combination_feature = combination_features[layer]
            style_loss_value = style_loss(style_feature, combination_feature)
            loss += style_loss_value * style_weight

    grads = tape.gradient(loss, combination_image)
    return loss, grads

if __name__ == "__main__":
    base_model = VGG19(weights="imagenet", include_top=False)
    outputs = dict([(layer.name, layer.output) for layer in base_model.layers])
    print(outputs)
    model = Model(inputs=base_model.input, outputs=outputs)
    content_layer_name = "block5_conv2"
    style_layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    total_variation_weight = 1e-6
    style_weight = 1e-6/len(style_layer_names)
    content_weight = 2.5e-7

    # content image
    base_image = cv2.imdecode(np.fromfile("dog.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w, _ = base_image.shape
    height = 300
    width = round(w*height/h)
    base_image = cv2.resize(base_image, (width, height))
    base_image = base_image[:,:,::-1].copy()
    base_image = np.expand_dims(base_image, axis=0)
    base_image = preprocess_input(base_image)
    print(base_image.shape)

    # style image
    style_image = cv2.imdecode(np.fromfile("vango.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
    style_image = cv2.resize(style_image, (width, height))
    style_image = style_image[:,:,::-1].copy()
    style_image = np.expand_dims(style_image, axis=0)
    style_image = preprocess_input(style_image)
    print(style_image.shape)

    combination_image = tf.Variable(base_image)
    output_path = "./output"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    epochs = 250
    optimizer = SGD(tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0,
        decay_steps=100,
        decay_rate=0.96
    ))

    fig, ax = plt.subplots()

    for epoch in range(epochs):
        loss, grads = compute_loss_and_grads(combination_image, base_image, style_image)
        optimizer.apply_gradients([(grads, combination_image)])
        print(f"Epoch: {epoch+1}")
        img = deprocess_image(combination_image.numpy())

        if epoch % 100 == 0:
            file_path = os.path.join(output_path, f"combination_epoch{epoch}.jpg")
            keras.utils.save_img(file_path, img)
            ax.clear()
            ax.axis(False)
            ax.imshow(img)
            plt.pause(0.1)
    plt.show()
