import tensorflow as tf
tfver = tf.__version__
print ("\n #### TF version: " + tfver + "\n")
print ("\n #### GPU available: " + str(len(tf.config.list_physical_devices('GPU'))) + " (" + tf.config.list_physical_devices('GPU')[0].name + ")\n")