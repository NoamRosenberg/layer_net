import os
import sys
import urllib
import tarfile
import tensorflow as tf

#add queues???

class cifarData:

	# Function for progress bar
	def _progress(self, count, block_size, total_size):
		sys.stdout.write('\r>> Downloading %s %.1f%%' % (
			self.tarfilename, 100.0 * count * block_size / total_size))
		sys.stdout.flush()

	# Function for parsing Cifar into a proper data set
	def _dataset_parser(self, value):

		# Every record consists of a label followed by the image, with a fixed number of bytes for each.
		label_bytes = 1
		image_bytes = 32 * 32 * 3
		record_bytes = label_bytes + image_bytes

		# Convert from a string to a vector of uint8 that is record_bytes long.
		raw_record = tf.decode_raw(value, tf.uint8)

		# The first byte represents the label, which we convert from uint8 to int32.
		label = tf.cast(raw_record[0], tf.int32)

		# The remaining bytes after the label represent the image, which we reshape
		# from [depth * height * width] to [depth, height, width].
		depth_major = tf.reshape(raw_record[label_bytes:record_bytes],
				               [3, 32, 32])

		# Convert from [depth, height, width] to [height, width, depth], and cast as
		# float32.
		image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

		return image, tf.one_hot(label, 10)

	#Function for augmenting images
	def _train_preprocess_fn(self, image, label):
		"""Preprocess a single training image of layout [height, width, depth]."""

		global rand_seed
		rand_seed = rand_seed + 1
		print('Using seed: ' + str(rand_seed))


		#image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
		image = tf.random_crop(image, [24, 24, 3], rand_seed)
		image = tf.image.random_flip_left_right(image, rand_seed)
		image = tf.image.random_brightness(image, 63, rand_seed)
		image = tf.image.random_contrast(image,0.2,1.8, rand_seed)
		image = tf.image.per_image_standardization(image)
		return image, label

	def _test_preprocess_fn(self, image, label):
		
		image = tf.image.resize_image_with_crop_or_pad(image, 24, 24)
		image = tf.image.per_image_standardization(image)
		return image, label

	def __init__(self):

		self.DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

	def load(self):

		# Download Cifar and extract if it doesn't exist allready
		data_dir = '/tmp/cifar10_data'
		filepath = os.path.join(data_dir,'cifar-10-batches-bin')
		if not os.path.exists(filepath):
			if not os.path.exists(data_dir):
				os.makedirs(data_dir)
			self.tarfilename = self.DATA_URL.split('/')[-1]
			tarfilepath = os.path.join(data_dir, self.tarfilename)

			tarfilepath, _ = urllib.request.urlretrieve(self.DATA_URL, tarfilepath, self._progress)
			print()
			statinfo = os.stat(tarfilepath)
			print('Successfully downloaded', self.tarfilename, statinfo.st_size, 'bytes.')
			tarfile.open(tarfilepath, 'r:gz').extractall(data_dir)

		filenames = [os.path.join(filepath, 'data_batch_%d.bin' % i) for i in range(1,6)]
		dataset = tf.data.FixedLengthRecordDataset(filenames,32 * 32 * 3 + 1)
		dataset = dataset.map(self._dataset_parser)
		global rand_seed
		rand_seed = 5000
		#250k samples for Shai
		#dataset = dataset.repeat(5)

		dataset = dataset.map(self._train_preprocess_fn)

		iterator = dataset.batch(250000).make_one_shot_iterator()
		data, labels = iterator.get_next()

		testfilename = [os.path.join(filepath,'test_batch.bin')]
		testdata = tf.data.FixedLengthRecordDataset(testfilename,32 * 32 * 3 + 1)
		testdata = testdata.map(self._dataset_parser)
		testdata = testdata.map(self._test_preprocess_fn)

		testiterator = testdata.batch(10000).make_one_shot_iterator()
		test_data, test_labels = testiterator.get_next()
		with tf.Session() as session:
			data, labels = session.run([data, labels])
			test_data, test_labels = session.run([test_data, test_labels])

		return data, test_data, labels, test_labels
