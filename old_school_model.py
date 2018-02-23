import tensorflow as tf
import numpy as np
from ops import linear, conv2d
from data import cifarData
from sklearn.metrics import accuracy_score



batch_size = 128
image_size = 32
c_dim = 3
classes=10

class Graph:

	def model(self, image, ver=3, reuse=False):
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

	def train(self, layer=9, epochs=2):

		data, test_data, labels, test_labels = self.allData

		for epoch in range(epochs):
			print("epoch: ", epoch + 1)
			for i in range(int(data.shape[0] / self.FLAGS.batch_size)):
				batch_data = data[i * self.FLAGS.batch_size:(i + 1) * self.FLAGS.batch_size]
				batch_labels = labels[i * self.FLAGS.batch_size:(i + 1) * self.FLAGS.batch_size]
				_, epoch_loss = self.sess.run([self.optim[layer], self.loss], feed_dict={
																	self.images: batch_data,
																	self.tags: batch_labels,
																	self.version: layer})
				if i % 500 == 0:
					print(epoch_loss)
			print(epoch_loss)
			nptest = self.sess.run(self.predictions, feed_dict={
												self.test_images:test_data,
												self.version: 9})
			pred_test = np.argmax(nptest,axis=1)
			y_test = np.argmax(test_labels, axis=1)
			print('accuracy score: ', accuracy_score(y_test,pred_test))

	def __init__(self, FLAGS):
		self.FLAGS = FLAGS
		cifar = cifarData()
		self.allData = cifar.load()	

	def run(self):
		self.num_epochs = self.FLAGS.epochs
		self.tags = tf.placeholder(tf.int32, [self.FLAGS.batch_size, classes],name='label')
		self.images = tf.placeholder(tf.float32, [self.FLAGS.batch_size, image_size, image_size, c_dim], name='image')
		self.version = tf.placeholder(tf.uint8,name='version')
		self.test_images = tf.placeholder(tf.float32, [10000, image_size, image_size, c_dim], name='image')

		nnlogits1, _ = self.model(self.images, self.version)
		self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.tags, logits=nnlogits1, scope='cross_entropy'),name='reduce_batch')
		nnlogits2, _ = self.model(self.test_images, self.version, reuse=True)
		self.predictions = tf.nn.softmax(nnlogits2)

		t_vars = tf.trainable_variables()
		h0_vars = [var for var in t_vars if 'h0' in var.name]
		h1_vars= [var for var in t_vars if 'h1' in var.name]
		h2_vars= [var for var in t_vars if 'h2' in var.name]
		h3_vars= [var for var in t_vars if 'h3' in var.name]

		optim0 = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss, var_list=h0_vars)
		optim1 = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss, var_list=h1_vars)
		optim2 = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss, var_list=h2_vars)
		optim3 = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss, var_list=h3_vars)
		regular_optim = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

		self.optim = {0: optim0, 1: optim1, 2: optim2, 3: optim3, 9: regular_optim}

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		if self.FLAGS.model_type == 'regular':
			self.train(layer=9, epochs=self.num_epochs)
		else:
			self.train(layer=0, epochs=int(self.num_epochs/4)+1)
			self.train(layer=1, epochs=int(self.num_epochs/4)+1)
			self.train(layer=2, epochs=int(self.num_epochs/4)+1)
			self.train(layer=3, epochs=int(self.num_epochs/4)+1)

		self.sess.close()
		




	
