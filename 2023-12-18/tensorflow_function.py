import tensorflow as tf

a = tf.square(tf.constant(10))
print(a)

a = tf.square(tf.constant([10, 9, 4]))
print(a)

a = tf.square([10, 9, 4])
print(a)