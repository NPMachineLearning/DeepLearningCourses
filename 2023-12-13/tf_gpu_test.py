import tensorflow as tf

gpus = tf.config.list_logical_devices(device_type="GPU")
print(gpus)