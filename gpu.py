import tensorflow as tf

# Detailed GPU info
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    details = tf.config.experimental.get_device_details(gpu)
    print(details)
    
print(tf.test.is_built_with_cuda())  # Should print True
print(tf.config.experimental.list_physical_devices('GPU'))  # Should list your GPU
print(tf.test.gpu_device_name())  # Should print something like /device:GPU:0
