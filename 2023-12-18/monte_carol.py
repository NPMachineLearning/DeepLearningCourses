import tensorflow as tf
import time

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))
print(tf.test.gpu_device_name())

batch = 100000000
epochs = 1000
incircle = 0

for epoch in range(epochs):
    t1 = time.time()
    X = tf.random.uniform([batch])
    y = tf.random.uniform([batch])
    dist = tf.sqrt(tf.square(X) + tf.square(y))
    incircle += tf.where(dist<=1.0).shape[0]
    area = incircle / ((epoch+1) * batch)
    pi = area * 4
    t2 = time.time()
    print(f"Epoch: {epoch} | PI: {pi} | Time: {t2 - t1:.5f} seconds")