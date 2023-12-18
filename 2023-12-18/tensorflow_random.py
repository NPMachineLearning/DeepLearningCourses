import tensorflow as tf
import pylab as plt

batch = 10000

X = tf.random.normal([batch])
y = tf.random.normal([batch])
plt.scatter(X, y, s=1)
plt.show()

X = tf.random.uniform([batch])
y = tf.random.uniform([batch])
plt.scatter(X, y, s=1)
plt.show()