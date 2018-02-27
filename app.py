import sys
import argparse
parser = argparse.ArgumentParser()
from old_school_model import Graph
from data import cifarData
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser.add_argument('--model_type', type=str, default='regular',
help='regular or layer model')

parser.add_argument('--epochs', type=int, default=20)

parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--dev', type=bool, default=False,
help='set dev to True to test that everythings working, in practice we only train on a single batch of the training data')

		
def main():
	cifar = cifarData()
	if FLAGS.dev == True:
		logger.info('training will be preformed on one single batch as part or the preformance test')
	logger.info('loading cifar10 . . .')
	allData = cifar.load()	
	graph = Graph(FLAGS, allData)
	logger.info('load graph and commence training')
	graph.run()


if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()
	main()

