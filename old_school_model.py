import tensorflow as tf
import numpy as np
from ops import linear, conv2d
from sklearn.metrics import accuracy_score
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


image_size = 24
c_dim = 3
classes=10
save_dir='save'


class Graph:

	def _model(self, image, ver=9, reuse=False):
		with tf.variable_scope("model") as scope:
			if reuse:
				scope.reuse_variables()

			h0_conv = conv2d(image,64,name='h0_conv')
			h0_activ =  tf.nn.relu(h0_conv, name='h0_activ')
			h0_pool = tf.nn.max_pool(h0_activ ,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME', name='h0_pool')
			h0_norm = tf.nn.lrn(h0_pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='h0_norm')
			if ver==0:
				h0_dense_ret = linear(tf.reshape(h0_norm, [image.shape.as_list()[0],-1]), 10, 'h0_dense_ret', stddev=0.05)
				layers = [h0_norm]
				return h0_dense_ret, layers

			h1_conv = conv2d(h0_norm,64, bias_init=0.1, name='h1_conv')
			h1_activ = tf.nn.relu(h1_conv, name='h1_activ')
			h1_norm = tf.nn.lrn(h1_activ, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='h1_norm')
			h1_pool = tf.nn.max_pool(h1_norm ,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME', name='h1_pool')
			if ver==1:
				h1_dense_ret = linear(tf.reshape(h1_pool, [image.shape.as_list()[0],-1]), 10, 'h1_dense_ret', stddev=5e-3)
				layers = [h0_norm, h1_pool]
				return h1_dense_ret, layers

			h2_dense = linear(tf.reshape(h1_pool, [image.shape.as_list()[0],-1]), 384,'h2_dense', stddev=0.04, bias_init=0.1, wd=0.004)
			h2_activ = tf.nn.relu(h2_dense, name='h2_activ')

			h3_dense = linear(h2_activ, 192,'h3_dense', stddev=0.04, bias_init=0.1, wd=0.004)
			h3_activ = tf.nn.relu(h3_dense, name='h3_activ')

			h4_dense = linear(h3_activ, 10, 'h3_dense_ret',stddev=5e-3)
			layers = [h0_pool, h1_pool, h2_activ, h3_activ]

			return h4_dense, layers

	def _train(self, layer=9, epochs=2):

		self.data, self.test_data, self.labels, self.test_labels = self.allData
		
		if self.FLAGS.dev == True:
			self.data = self.data[:128]
			self.labels = self.labels[:128]

		for epoch in range(epochs):
			print("layer: ", layer, "epoch: ", epoch + 1)
			for i in range(int(self.data.shape[0] / self.FLAGS.batch_size)):
				batch_data = self.data[i * self.FLAGS.batch_size:(i + 1) * self.FLAGS.batch_size]
				batch_labels = self.labels[i * self.FLAGS.batch_size:(i + 1) * self.FLAGS.batch_size]
				_, epoch_loss, l_rate = self.sess.run([self.optim[layer], self.loss, self.lr], feed_dict={
																	self.images: batch_data,
																	self.tags: batch_labels,
																	self.version: layer})
				if i % 500 == 0:
					print('loss: ',epoch_loss,'  learning rate: ', l_rate)

			nptest = self.sess.run(self.predictions, feed_dict={
												self.test_images:self.test_data,
												self.version: layer})
			pred_test = np.argmax(nptest,axis=1)
			y_test = np.argmax(self.test_labels, axis=1)
			acc_score = accuracy_score(y_test,pred_test)
			print('val accuracy score: ', acc_score)
			self.test_accuracy_improvements.append(acc_score)						

	def _write_accuracy_improvements(self,acc):
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		np.savetxt(save_dir	+ '/accuracy_improvements.csv', acc, delimiter=',')

	def _writeTestNeurons(self, neurons, ver, data, labels):
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		for j in range(len(neurons)):
			t_path = save_dir + '/t_test' + str(j) + '.csv'
			l_path = save_dir + '/l_test' + str(j) + '.csv'
			with open(t_path, 'wb') as t:
				with open(l_path, 'wb') as l:
					units = self.sess.run(neurons[j], feed_dict={
														self.test_images: data,
														self.version: ver})
					flat_units = units.reshape((units.shape[0],-1))
					np.savetxt(t,flat_units, delimiter=',', fmt='%.8e')
					np.savetxt(l,labels, delimiter=',', fmt='%.8e')

	def _writeTrainNeurons(self, neurons, ver, data, labels):
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		for j in range(len(neurons)):
			t_path = save_dir + '/t_train' + str(j) + '.csv'
			l_path = save_dir + '/l_train' + str(j) + '.csv'
			with open(t_path, 'wb') as t:
				with open(l_path, 'wb') as l:
					for i in range(int(data.shape[0] / self.FLAGS.batch_size)):
						batch_data = data[i * self.FLAGS.batch_size:(i + 1) * self.FLAGS.batch_size]
						units = self.sess.run(neurons[j], feed_dict={
															self.images: batch_data,
															self.version: ver})
						flat_units = units.reshape((units.shape[0],-1))
						np.savetxt(t,flat_units, delimiter=',', fmt='%.8e')
						np.savetxt(l,labels, delimiter=',', fmt='%.8e')

	def __init__(self, FLAGS, allData):

		self.FLAGS = FLAGS
		self.allData = allData	

	def run(self):

		self.num_epochs = self.FLAGS.epochs
		self.tags = tf.placeholder(tf.int32, [self.FLAGS.batch_size, classes],name='label')
		self.images = tf.placeholder(tf.float32, [self.FLAGS.batch_size, image_size, image_size, c_dim], name='image')
		self.version = tf.placeholder(tf.uint8,name='version')
		self.test_images = tf.placeholder(tf.float32, [10000, image_size, image_size, c_dim], name='image')

		nnlogits1, neurons1 = self._model(self.images, self.version)

		self.softmax_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.tags, logits=nnlogits1, scope='cross_entropy'),name='reduce_batch')
		self.loss = self.softmax_loss + sum(tf.get_collection('regularizations'))

		nnlogits2, neurons2 = self._model(self.test_images, self.version, reuse=True)
		self.predictions = tf.nn.softmax(nnlogits2)

		t_vars = tf.trainable_variables()
		h0_vars = [var for var in t_vars if 'h0' in var.name]
		h1_vars= [var for var in t_vars if 'h1' in var.name]

		num_batches_per_epoch = self.allData[0].shape[0] / self.FLAGS.batch_size
		decay_steps = int(num_batches_per_epoch * self.FLAGS.num_epochs_per_decay)
		global_step = tf.Variable(0, trainable=False, name='global_step')
		global_step0 = tf.Variable(0, trainable=False, name='global_step0')
		global_step1 = tf.Variable(0, trainable=False, name='global_step1')
		self.lr = tf.train.exponential_decay(0.1,global_step,decay_steps, self.FLAGS.lr_decay_factor,staircase=True)

		regular_optim = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=global_step)
		optim0 = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=global_step0, var_list=h0_vars)
		optim1 = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=global_step1, var_list=h1_vars)

		self.optim = {0: optim0, 1: optim1, 9: regular_optim}

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		self.test_accuracy_improvements = []		
		if self.FLAGS.model_type == 'regular':
			lastversion = 9
			self._train(layer=lastversion, epochs=self.num_epochs)
		elif self.FLAGS.model_type == 'first_two':
			print('layer 0 training')			
			self._train(layer=0, epochs=20)
			lastversion = 1
			print('layer ',lastversion,' training')
			self._train(layer=lastversion, epochs=20)
		else:
			print('layer 0 training')			
			self._train(layer=0, epochs=20)
			print('layer 1 training')
			self._train(layer=1, epochs=20)
			lastversion = 9
			print('layer ',lastversion,' training')
			self._train(layer=lastversion, epochs=self.num_epochs)
		logger.info('writing accuracy improvements to the save file')
		self._write_accuracy_improvements(self.test_accuracy_improvements)
		logger.info('Preparing to write train and test neurons to the save file, this may take some time and require lots of disc space')
		if self.FLAGS.model_type != 'regular':
			self._writeTrainNeurons(neurons1, lastversion, self.data, self.labels)
			self._writeTestNeurons(neurons2, lastversion, self.test_data, self.test_labels)

		self.sess.close()




	
