import tensorflow as tf
import numpy as np
from ops import linear, conv2d
from data import load_data
from sklearn.metrics import accuracy_score
import sys
import argparse
parser = argparse.ArgumentParser()


batch_size = 128
image_size = 32
c_dim = 3
classes=10

parser.add_argument('--model_type', type=str, default='regular',
help='regular or layer model')

parser.add_argument('--epochs', type=int, default=20,
help='apx sum epochs for training')

def model(image, ver=3, reuse=False):
	with tf.variable_scope("model") as scope:
		if reuse:
			scope.reuse_variables()

		h0_conv = conv2d(image,64,name='h0_conv')
		h0_activ =  tf.nn.relu(h0_conv, name='h0_activ')
		h0_pool = tf.nn.max_pool(h0_activ ,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME', name='h0_pool')
		if ver==0:
			return linear(tf.reshape(h0_pool, [image.shape.as_list()[0],-1]), 10, 'h0_dense_ret'), [1]

		h1_conv = conv2d(h0_pool,64,name='h1_conv')
		h1_activ = tf.nn.relu(h1_conv, name='h1_activ')
		h1_pool = tf.nn.max_pool(h1_activ ,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME', name='h1_pool')
		if ver==1:
			return linear(tf.reshape(h1_pool, [image.shape.as_list()[0],-1]), 10, 'h1_dense_ret'), [1]

		h2_dense = linear(tf.reshape(h1_pool, [image.shape.as_list()[0],-1]), 384,'h2_dense')
		h2_activ = tf.nn.relu(h2_dense, name='h2_activ')
		if ver==2:
			return linear(h2_activ, 10, 'h2_dense_ret'), [1]

		h3_dense = linear(h2_activ, 192,'h3_dense')
		h3_activ = tf.nn.relu(h3_dense, name='h3_activ')

		h4_dense = linear(h3_activ, 10, 'h3_dense_ret')
		layers = [h0_conv, h0_activ, h0_pool,h1_conv, h1_activ, h1_pool, h2_dense, h2_activ, h3_dense, h3_activ, h4_dense]

		return h4_dense, layers

def train(layer=9, epochs=2):

	for epoch in range(epochs):
		print("epoch: ", epoch + 1)
		for i in range(int(data.shape[0] / batch_size)):
			batch_data = data[i * batch_size:(i + 1) * batch_size]
			batch_labels = labels[i * batch_size:(i + 1) * batch_size]
			_, epoch_loss = sess.run([optim[layer], loss], feed_dict={
																images: batch_data,
																tags: batch_labels,
																version: layer})
			if i % 500 == 0:
				print(epoch_loss)
		print(epoch_loss)
		nptest = sess.run(predictions, feed_dict={
											test_images:test_data,
											version: 9})
		pred_test = np.argmax(nptest,axis=1)
		y_test = np.argmax(test_labels, axis=1)
		print('accuracy score: ', accuracy_score(y_test,pred_test))	

if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()

	data, test_data, labels, test_labels = load_data()

	tags = tf.placeholder(tf.int32, [batch_size, classes],name='label')
	images = tf.placeholder(tf.float32, [batch_size, image_size, image_size, c_dim], name='image')
	version = tf.placeholder(tf.uint8,name='version')
	test_images = tf.placeholder(tf.float32, [10000, image_size, image_size, c_dim], name='image')

	nnlogits1, _ = model(images, version)
	loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=tags, logits=nnlogits1, scope='cross_entropy'),name='reduce_batch')

	t_vars = tf.trainable_variables()
	h0_vars = [var for var in t_vars if 'h0' in var.name]
	h1_vars= [var for var in t_vars if 'h1' in var.name]
	h2_vars= [var for var in t_vars if 'h2' in var.name]
	h3_vars= [var for var in t_vars if 'h3' in var.name]

	optim0 = tf.train.GradientDescentOptimizer(0.01).minimize(loss, var_list=h0_vars)
	optim1 = tf.train.GradientDescentOptimizer(0.01).minimize(loss, var_list=h1_vars)
	optim2 = tf.train.GradientDescentOptimizer(0.01).minimize(loss, var_list=h2_vars)
	optim3 = tf.train.GradientDescentOptimizer(0.01).minimize(loss, var_list=h3_vars)
	regular_optim = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
	optim = {0: optim0, 1: optim1, 2: optim2, 3: optim3, 9: regular_optim}
	global sess
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		num_epochs = FLAGS.epochs
		if FLAGS.model_type == 'regular':
			train(layer=9, epochs=num_epochs)
		else:
			train(layer=0, epochs=int(num_epochs/4)+1)
			train(layer=1, epochs=int(num_epochs/4)+1)
			train(layer=2, epochs=int(num_epochs/4)+1)
			train(layer=3, epochs=int(num_epochs/4)+1)



	
