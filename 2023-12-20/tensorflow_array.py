import os
import tensorflow as tf
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# create 100 zeros array
zeros = tf.zeros(100)
print(zeros)
print(tf.zeros([100]))

# create 100 ones array
ones = tf.ones(100)
print(ones)
print(tf.ones([100]))

# create 3x10 dimension array
arr = tf.zeros([3, 10])
print(arr)

# fill array with value
arr = tf.fill([10], 5)
print(arr)

# range of values
arr = tf.range(0, 9)
print(arr)
print(tf.range(0, 9, 2)) # step by 2

# normal
normal = tf.random.normal([10])
print(normal)