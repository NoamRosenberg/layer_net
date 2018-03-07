import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


batch_size = 128
image_size = 24
c_dim = 3
classes=10

def conv2d(input_, output_dim, name="conv2d"):
  with tf.variable_scope(name):

    w = tf.get_variable('w', [5, 5, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=0.02))

    conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


def linear(input_, output_size, scope=None, stddev=0.02, with_w=False):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):

    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=0.02))

    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(0.0))

    return tf.matmul(input_, matrix) + bias

def _model(image, reuse=False):
	with tf.variable_scope("model") as scope:
		if reuse:
			scope.reuse_variables()

		h0_conv = conv2d(image,64,name='h0_conv')
		h0_activ =  tf.nn.relu(h0_conv, name='h0_activ')
		h0_pool = tf.nn.max_pool(h0_activ ,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME', name='h0_pool')
		h0_norm = tf.nn.lrn(h0_pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='h0_norm')
		h1_conv = conv2d(h0_norm,64,name='h1_conv')
		h1_activ = tf.nn.relu(h1_conv, name='h1_activ')
		h1_norm = tf.nn.lrn(h1_activ, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='h1_norm')
		h1_pool = tf.nn.max_pool(h1_norm ,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME', name='h1_pool')
		h2_dense = linear(tf.reshape(h1_pool, [image.shape.as_list()[0],-1]), 384,'h2_dense')
		h2_activ = tf.nn.relu(h2_dense, name='h2_activ')
		h3_dense = linear(h2_activ, 192,'h3_dense')
		h3_activ = tf.nn.relu(h3_dense, name='h3_activ')

		h4_dense = linear(h3_activ, 10, 'h3_dense_ret')

		return h4_dense

tags = tf.placeholder(tf.int32, [batch_size, classes],name='label')
images = tf.placeholder(tf.float32, [batch_size, image_size, image_size, c_dim], name='image')
version = tf.placeholder(tf.uint8,name='version')

nnlogits = _model(images)


