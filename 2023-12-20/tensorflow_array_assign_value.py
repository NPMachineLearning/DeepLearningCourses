import tensorflow as tf

a = tf.zeros([10])
print(a)
a = tf.Variable(a)
a[1].assign(12)
print(a)