import tensorflow as tf

def conv2d(input_, output_dim, name="conv2d"):
  with tf.variable_scope(name):

    w = tf.get_variable('w', [5, 5, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=0.05))

    conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


def linear(input_, output_size, scope=None, stddev=0.02, wd=None):
	shape = input_.get_shape().as_list()
	with tf.variable_scope(scope or "Linear"):

		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=0.02))

		if wd is not None:
			tf.add_to_collection('regularizations',wd * tf.nn.l2_loss(matrix))

		bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(0.0))

		return tf.matmul(input_, matrix) + bias
