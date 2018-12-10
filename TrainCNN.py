import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import utils.network_utils as net_utils
from utils.display import APDDisplay

def parse_args(args, flags):
	flags.DEFINE_string('optimizer', args.optimizer, 'Name of the minimization optimizer')
	flags.DEFINE_integer('numEpochs', args.numEpochs, 'Number of epochs to train for')
	flags.DEFINE_integer('batchSize', args.batchSize, 'Batch size')
	flags.DEFINE_integer('maxTrainSteps', args.maxTrainSteps, 'Maximum number of steps to train')
	flags.DEFINE_float('learningRate', args.learningRate, 'Learning rate')
	flags.DEFINE_float('regTerm', args.regTerm, 'Regularization parameter')
	flags.DEFINE_float('dropout', args.dropout, 'Keep probability for the dropout layer')
	flags.DEFINE_string('outDir', args.outDir, 'Output directory')
	flags.DEFINE_string('model', args.model, 'Model name if training is resumed')
	flags.DEFINE_boolean('trainEnergy', args.trainEnergy, 'Train on energy')
	flags.DEFINE_string('trainingSet', args.trainingSet, 'Filename of the training set')
	flags.DEFINE_string('testSet', args.testSet, 'Filename of the test set')

	return

def get_dataset(filename, flags):
	nFeatures = 74*350
	nLabels = 5

	record_defaults = [[0.0]]*(nFeatures + nLabels)
	dataset = tf.data.experimental.CsvDataset(filename, record_defaults) \
			.shuffle(buffer_size=1000) \
			.batch(flags.batchSize) \
			.repeat(flags.numEpochs)

	iterator = dataset.make_one_shot_iterator()
	x = iterator.get_next()

	if flags.trainEnergy:
		return x[0:nFeatures], x[nFeatures+3:nFeatures+4]
	else:
		return x[0:nFeatures], x[nFeatures:nFeatures+3]

def main(args):
	flags = tf.app.flags
	FLAGS = flags.FLAGS

	parse_args(args, flags)

	# Create network

	# Placeholder for input image
	x = tf.placeholder(tf.float32, shape=[None, 74*350])
	x_image = tf.reshape(x, [-1, 74, 350, 1])

	# Placeholder for output
	nOutput = 3
	if FLAGS.trainEnergy:
		nOutput = 1
	y_ = tf.placeholder(tf.float32, shape=[None, nOutput])

	# Placeholder for regularization
	regularization = tf.placeholder(tf.float32)

	y, reg_l2 = net_utils.build_network(x_image, FLAGS)

	# Loss function
	mse = tf.reduce_mean(tf.square(y-y_))
	loss = mse + regularization*reg_l2

	# Build optimizer
	if FLAGS.optimizer == 'Gradient':	
		opt = tf.train.GradientDescentOptimizer(FLAGS.learningRate)
	elif FLAGS.optimizer == 'Adam':
		opt = tf.train.AdamOptimizer(FLAGS.learningRate)
	
	global_step = tf.train.get_or_create_global_step()
	train_step = opt.minimize(loss, global_step=global_step)

	# Create dataset
	data_train_x, data_train_y = get_dataset(FLAGS.trainingSet, FLAGS)
	data_val_x, data_val_y = get_dataset(FLAGS.testSet, FLAGS)

	hooks=[tf.train.StopAtStepHook(last_step=FLAGS.maxTrainSteps)]

	with tf.train.MonitoredTrainingSession(
						master='',
						is_chief=True,
						checkpoint_dir='./logs/',
						hooks=hooks) as sess:
		while not sess.should_stop():
			batch_x, batch_y = sess.run([data_train_x, data_train_y])
			current_loss, _ = sess.run([mse, train_step], feed_dict={x: np.transpose(batch_x), y_: np.transpose(batch_y), regularization: FLAGS.regTerm})

			print('loss = %f'%current_loss)

		sess.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--optimizer', type=str, default='Adam')
	parser.add_argument('--numEpochs', type=int, default=None)
	parser.add_argument('--batchSize', type=int, default=500)
	parser.add_argument('--maxTrainSteps', type=int, default=10000)
	parser.add_argument('--learningRate', type=float, default=0.001)
	parser.add_argument('--regTerm', type=float, default=1e-7)
	parser.add_argument('--dropout', type=float, default=1.0)
	parser.add_argument('--outDir', type=str, default='./output/')
	parser.add_argument('--model', type=str, default='')
	parser.add_argument('--trainEnergy', action='store_true')
	parser.add_argument('--evalPlotName', type=str, default='')
	parser.add_argument('--evalFilename', type=str, default='')
	parser.add_argument('--saveRawFile', action='store_true')
	parser.add_argument('--trainingSet', type=str, default='')
	parser.add_argument('--testSet', type=str, default='')
	parser.add_argument('--savePredPlots', action='store_true')

	args = parser.parse_args()

	main(args)
