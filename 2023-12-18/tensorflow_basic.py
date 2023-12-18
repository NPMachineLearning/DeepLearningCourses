import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# constant variable immutable
tensor = tf.constant("Hello Tensor")
print(tensor)
print(tensor.numpy())
print(hex(id(tensor)))
# error no method tf.assing(2)

# create another tensor object
tensor = tf.constant(5)
print(tensor)
print(hex(id(tensor)))

# create tensor variable which is mutable
tensor_var = tf.Variable(10)
print(tensor_var)
print(hex(id(tensor_var)))
# udpate tensor value
tensor_var.assign(2)
print(tensor_var)
print(hex(id(tensor_var)))

# tensor operation
a = tf.constant(5)
b = tf.Variable(5)
print(a, b)
print(a + b)
print(a * b)
print(a / b)
print(a // b)
print(a % b)

af = tf.constant(5.)
bf = tf.Variable(5.)
print(af, bf)
print(af + bf)
print(af * bf)
print(af / bf)
print(af // bf)
# error print(af % bf)

# python with tensor
print(a/2)

# type conversion
a = tf.constant(10.5223, dtype=tf.float32)
print(a)
a_converted = tf.cast(a, dtype=tf.int32)
print(a_converted)

# random number
