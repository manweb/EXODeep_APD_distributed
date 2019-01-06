import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
import utils.network_utils as net_utils
from utils.cluster_utils import setup_slurm_cluster
from utils.display import APDDisplay

def parse_args(flags):
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
	parser.add_argument('--numPS', type=int, default=1)

	args = parser.parse_args()

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
	flags.DEFINE_integer('numPS', args.numPS, 'Number of parameter servers')

	print("#################################################")
	print("  Training neural network with")
	for k, v in flags.FLAGS.flag_values_dict().items():
		print('\t%s:\t\t%s'%(k, str(v)))
	print("#################################################")

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

def main(_):
	flags = tf.app.flags
	FLAGS = flags.FLAGS

	flags.DEFINE_string('job_name', None, 'job name: worker or ps')
	flags.DEFINE_integer('task_index', 0, 'Worker task index, should be >= 0. task_index = 0 is the chief worker task which performs the variable initialiazation')

	parse_args(flags)

	# Create cluster specification
	cluster, server, FLAGS.task_index, num_tasks, FLAGS.job_name = setup_slurm_cluster(num_ps = FLAGS.numPS)

	is_chief = (FLAGS.job_name == 'worker' and FLAGS.task_index == 0)

	if cluster and is_chief:
		print('Performing distributed training')
	elif is_chief:
		print('Performing single machine training')

	if FLAGS.job_name == 'ps':
		server.join()
	elif FLAGS.job_name == 'worker':
		if cluster:
			worker_device = '/job:worker/task:%d'%FLAGS.task_index
			device = tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster)
			target = server.target
		else:
			device = None
			target = ''

		# Create network
		with tf.device(device):
	
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
		
			y, reg_l2 = net_utils.build_network(x_image, FLAGS, is_chief)
		
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

		lossName = '%s/loss_%.7f_%.7f.csv'%(FLAGS.outDir, FLAGS.regTerm, FLAGS.learningRate)
	
		with tf.train.MonitoredTrainingSession(
							master=target,
							is_chief=(FLAGS.task_index==0),
							checkpoint_dir='%s/checkpoint/'%FLAGS.outDir,
							hooks=hooks) as sess:
			local_step = 0
			totalTime = time.time()
			while not sess.should_stop():
				try:
					batch_x, batch_y = sess.run([data_train_x, data_train_y])
					current_loss, glob_step, _ = sess.run([mse, global_step, train_step], feed_dict={x: np.transpose(batch_x), y_: np.transpose(batch_y), regularization: FLAGS.regTerm})
	
					if glob_step%10 == 0:
						batch_val_x, batch_val_y = sess.run([data_val_x, data_val_y])
						val_loss = sess.run(mse, feed_dict={x: np.transpose(batch_val_x), y_: np.transpose(batch_val_y)})
	
						#option = 'w' if local_step == 0 else 'a'
						option = 'a'
	
						f_out_loss = open(lossName, option)
						f_out_loss.write(','.join(np.char.mod('%f', np.array([local_step, glob_step, current_loss, val_loss])))+'\n')
						f_out_loss.close()
	
						print('local_step: %i, global_step: %i, worker_task: %i, train loss = %f, validation loss = %f'%(local_step, glob_step, FLAGS.task_index, current_loss, val_loss))
	
					local_step += 1
				except RuntimeError:
					break

			totalTime = time.time() - totalTime

			if is_chief:
				print('Done training. Total time = %fs'%totalTime)

if __name__ == '__main__':
	tf.app.run(main=main)
