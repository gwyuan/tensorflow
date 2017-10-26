import os
import os.path
import tensorflow as tf


INCEPTION_LOG_DIR = '/tmp/inception_v3_log'

if not os.path.exists(INCEPTION_LOG_DIR):
    os.makedirs(INCEPTION_LOG_DIR)
with tf.Session() as sess:
    model_filename = 'model/classify_image_graph_def.pb'
    with tf.gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    writer = tf.summary.FileWriter(INCEPTION_LOG_DIR, graph_def)
    writer.close()