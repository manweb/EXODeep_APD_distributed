import numpy as np
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle as pk
import utils.network_utils as net_utils
from utils.display import APDDisplay

def parse_args(flags):
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputFile', type=str)
	parser.add_argument('--model', type=str)
	parser.add_argument('--trainEnergy', action='store_true')
	parser.add_argument('--evalPlotName', type=str, default='')
	parser.add_argument('--scaleLabels', action='store_true')

	args = parser.parse_args()

	flags.DEFINE_string('inputFile', args.inputFile, 'Input filename')
	flags.DEFINE_string('model', args.model, 'Model filename')
	flags.DEFINE_boolean('trainEnergy', args.trainEnergy, 'Train on energy')
	flags.DEFINE_string('evalPlotName', args.evalPlotName, 'Filename of output')
	flags.DEFINE_boolean('scaleLabels', args.scaleLabels, 'Scale labels')
	flags.DEFINE_float('dropout', 1.0, 'Keep probability for the dropout layer')

	return

def Eval(_):
	flags = tf.app.flags
	FLAGS = flags.FLAGS

	parse_args(flags)

	x = tf.placeholder(tf.float32, shape=[None, 74*350])
	x_image = tf.reshape(x, [-1, 74, 350, 1])

	nOutput = 3
	if FLAGS.trainEnergy:
		nOutput = 1
	y_ = tf.placeholder(tf.float32, shape=[None, nOutput])

	y, _ = net_utils.build_network(x_image, FLAGS, True, FLAGS.model)

	if FLAGS.trainEnergy:
		mean_error = tf.reduce_mean(tf.abs(y-y_)/y_)
	else:
		mean_error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y-y_),1)))

	trueData = np.empty((0,nOutput), float)
	predictedData = np.empty((0,nOutput), float)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		with open(FLAGS.inputFile) as infile:
			count = 0
			sum_mean_error = 0
			nOutliers = 0
			for line in infile:
				im = np.fromstring(line, dtype=np.float32, sep=',')

				img = np.reshape(im[0:74*350], (1,74*350))

				if FLAGS.trainEnergy:
					labels = np.reshape(im[74*350+3:74*350+4], (1,1))
				else:
					labels = np.reshape(im[74*350:74*350+3], (1,3))

				predicted, mean_dist = sess.run([y, mean_error], feed_dict={x: img, y_: labels})

				#print('predicted: (%.2f,%.2f,%.2f), true: (%.2f,%.2f,%.2f)'%(prediction[0,0],prediction[0,1],prediction[0,2],labels[0],labels[1],labels[2]center = [100,0,100]

				trueData = np.append(trueData, np.array(labels), axis=0)
				predictedData = np.append(predictedData, np.array(predicted), axis=0)

				center = [100,0,100]
				box_size = 50

				#if predicted[0,0] < center[0]-box_size or predicted[0,0] > center[0]+box_size or predicted[0,1] < center[1]-box_size or predicted[0,1] > center[1]+box_size or predicted[0,2] < center[2]-box_size or predicted[0,2] > center[2]+box_size:
				#	option = 'a'
				#	if nOutliers == 0:
				#		option = 'w'

				#	f = open('outliers.csv', option)
				#	f.write(",".join(np.char.mod('%f', im))+'\n')
				#	f.close()

				#	nOutliers += 1

				#	print('Number of outliers: %i'%nOutliers)

				count += 1
				sum_mean_error += mean_dist

				#if count > 10:
				#	break

		sess.close()

		print('Number of events processed: %i'%count)

		sum_mean_error /= count

		unit = '%' if FLAGS.trainEnergy else "mm"
		scale = 100 if FLAGS.trainEnergy else 1
		print("mean distance = %.2f %s"%(sum_mean_error*scale,unit))
		print("mean = %.2f %s"%(np.mean(np.sqrt(np.sum(np.square(predictedData-trueData), axis=1))),unit))
		print("mean x = %.2f %s"%(np.mean(np.absolute(predictedData[:,0]-trueData[:,0])),unit))
		print("mean y = %.2f %s"%(np.mean(np.absolute(predictedData[:,1]-trueData[:,1])),unit))
		print("mean z = %.2f %s"%(np.mean(np.absolute(predictedData[:,2]-trueData[:,2])),unit))

                #title = "mean error = %.2f %s"%(sum_mean_error*scale,unit)
		title = ''

		dsp = APDDisplay()
		dsp.PlotHistos(predictedData, trueData, title, FLAGS.evalPlotName, FLAGS.trainEnergy, FLAGS.inputFile)
		dsp.DisplayPosition(predictedData, trueData, False, FLAGS.evalPlotName.replace('.','_2D.'), title, FLAGS.scaleLabels)
		#dsp.PlotPositionDiff(predictedData, trueData, evalPlotName.replace('.','_posDiff.'))

if __name__ == '__main__':
	tf.app.run(main=Eval)
	
