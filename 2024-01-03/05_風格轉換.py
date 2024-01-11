import os
import shutil

import cv2
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.optimizers import SGD

os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
import tensorflow as tf
#底下是逆向工程的作法
def deprocess_image(img):
    #調整圖片亮度
    img=img.reshape((height, width, 3))[:,:,::-1].copy()
    #底下依不同的圖片有不同的值
    # img[:, :, 0] += 123.67
    # img[:, :, 1] += 116.779
    # img[:, :, 2] += 103.939
    #fixed by Thomas - 2024/01/03 : 最後轉 uint8, 所以加小數沒意義
    img[:, :, 0] += 123
    img[:, :, 1] += 116
    img[:, :, 2] += 103
    #小於0就變為0，大於255就改為255
    return np.clip(img, 0,255).astype(np.uint8)
def gram_matrix(x):
    x=tf.reshape(x, (x.shape[1], x.shape[2], x.shape[3]))
    x=tf.transpose(x, (2, 0, 1))
    features=tf.reshape(x, (tf.shape(x)[0], -1))
    gram=tf.matmul(features, tf.transpose(features))
    return gram
def style_loss(style_feature, combination_feature):
    s = gram_matrix(style_feature)
    c = gram_matrix(combination_feature)
    channels=3
    size=width*height
    #(4 * (channels ** 2) * (size ** 2))是官網建議的，可以自已調整看看，可以看成是一個常數
    return tf.reduce_sum(tf.square(s-c))/(4*(channels**2)*(size**2))
def compute_loss_and_grads(combination_image, base_image, style_image):
    #計算合成照損失函數及梯度下降
    with tf.GradientTape() as tape:
        base_features=model(base_image)
        base_feature=base_features[content_layer_name]
        loss=tf.zeros(shape=())
        combination_features=model(combination_image)
        combination_feature=combination_features[content_layer_name]
        #損失函數 : 預測值及實際值的殘差平方總合
        #計算合成與內容 block5_conv2 的損失值
        loss=loss+tf.reduce_sum(tf.square(combination_feature-base_feature))*content_weight

        #取得風格5層特徵
        style_features=model(style_image)
        for layer in style_layer_names:
            style_feature=style_features[layer]
            combination_feature=combination_features[layer]
            #底下沒有經過格拉姆轉換
            #loss+= tf.reduce_sum(tf.square(style_feature-combination_feature)) * (1e-8 / len(style_layer_names))

            #底下的 style_loss 有經過格拉姆轉換
            loss += style_loss(style_feature, combination_feature)*style_weight
    grads=tape.gradient(loss, combination_image)
    return loss, grads
if __name__=="__main__":
    model_base=VGG19(weights="imagenet", include_top=False)
    outputs=dict([(layer.name, layer.output) for layer in model_base.layers])
    model=Model(inputs=model_base.inputs, outputs=outputs)
    content_layer_name='block5_conv2'
    style_layer_names=['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
    total_variation_weight=1e-6#總變異數
    style_weight=1e-6/len(style_layer_names)#風格權重
    content_weight=2.5e-8#原始內容權重

    #原始內容圖
    base_image=cv2.imdecode(np.fromfile('dog.jpg', dtype=np.uint8), cv2.IMREAD_COLOR)
    h,w,_=base_image.shape
    height = 600
    #height = 950
    height = 70
    width=round(w*height/h)
    base_image=cv2.resize(base_image, (width, height), interpolation=cv2.INTER_LINEAR)
    base_image=base_image[:,:,::-1].copy()
    base_image=np.expand_dims(base_image, axis=0)
    base_image=preprocess_input(base_image)

    #風格圖片
    style_image=cv2.imdecode(np.fromfile('vango.jpg', dtype=np.uint8), cv2.IMREAD_COLOR)
    style_image=cv2.resize(style_image, (width, height), interpolation=cv2.INTER_LINEAR)
    style_image=style_image[:,:,::-1].copy()
    style_image=np.expand_dims(style_image, axis=0)
    style_image=preprocess_input(style_image)

    combination_image=tf.Variable(base_image)
    output_path="./output"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    epochs=4000
    #優化器，計算損失函數的方法，SGD其實是指 MBGD
    #https://mahaljsp.ddns.net/normal_loss_rate/
    optimizer=SGD(
        tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=100.0,
            decay_steps=100,
            decay_rate=0.96
        )
    )
    fig, ax=plt.subplots()
    #開始進行4000次的合成照
    for i in range(epochs):
        loss, grads=compute_loss_and_grads(combination_image, base_image, style_image)
        optimizer.apply_gradients([(grads, combination_image)])
        print(f"eoch : {i+1}")
        img=deprocess_image(combination_image.numpy())#將合成照的亮度調亮
        if (i+1) %100 ==0:
            file=os.path.join(output_path, f'combination_epoch{i+1}.jpg')
            keras.utils.save_img(file, img)
        ax.clear()
        ax.axis("off")
        ax.imshow(img)
        plt.pause(0.01)
    plt.show()