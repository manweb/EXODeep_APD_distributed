import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import os
import pickle as pk
from utilities import DataSet
from display import APDDisplay
from tensorflow.python import pywrap_tensorflow
from architecture_medium import *

# Definition of the network architecture

# Name of the weight variables for each layer
#weightName = {  0: 'WConv0',
#                1: 'WConv1',
#                2: 'WConv2',
#                3: 'W0',
#                4: 'W1',
#                5: 'W2',
#                6: 'W3',
#                7: 'W4'}

# Name of the bias variable for each layer
#biasName = {    0: 'Variable',
#                1: 'Variable_1',
#                2: 'Variable_2',
#                3: 'Variable_3',
#                4: 'Variable_4',
#                5: 'Variable_5',
#                6: 'Variable_6',
#                7: 'Variable_7'}

# type of the layer
#layerType = {   0: 'conv',
#                1: 'conv',
#                2: 'conv',
#                3: 'fc',
#                4: 'fc',
#                5: 'fc',
#                6: 'fc',
#                7: 'fc'}

def Eval(filename, model, evalPlotName, trainEnergy = False, scaleLabels = False):
	# Retrieve model weights and recreate network
	reader = pywrap_tensorflow.NewCheckpointReader(model)

	x = tf.placeholder(tf.float32, shape=[None, 74*350])
        x_image = tf.reshape(x, [-1, 74, 350, 1])

	nOutput = 3
	if trainEnergy:
		nOutput = 1
	y_ = tf.placeholder(tf.float32, shape=[None, nOutput])

	nLayers = len(layerType)
	oldLayer = layerType[0]
	for layer in range(nLayers):
		weights = reader.get_tensor(weightName[layer])
                bias = reader.get_tensor(biasName[layer])

		print('Layer %i, type %s'%(layer,layerType[layer]))
		print('    w: %s'%str(weights.shape))
		print('    b: %s'%str(bias.shape))
		print('    max_pool: %s'%str(maxPoolingSize[layer]))

		if not layerType[layer] == oldLayer:
			H_pool = tf.reshape(H_pool, [-1, weights.shape[0]])

		if layer == 0 and layerType[layer] == 'conv':
                        H = tf.nn.relu(tf.nn.conv2d(x_image, weights, strides=[1,1,1,1], padding='SAME') + bias)
                elif layerType[layer] == 'conv':
                        H = tf.nn.relu(tf.nn.conv2d(H_pool, weights, strides=[1,1,1,1], padding='SAME') + bias)
		elif layer < nLayers-1:
			H = tf.nn.relu(tf.matmul(H_pool,weights)+bias)
		else:
			H = tf.matmul(H_pool,weights)+bias

		if layerType[layer] == 'conv':
			H_pool = tf.nn.max_pool(H, ksize=maxPoolingSize[layer], strides=maxPoolingSize[layer], padding='SAME')
		else:
			H_pool = H

		oldLayer = layerType[layer]

	if trainEnergy:
		mean_error = tf.reduce_mean(tf.abs(H_pool-y_)/y_)
	else:
		mean_error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(H_pool-y_),1)))

	if trainEnergy:
                trueData = np.empty((0,1), float)
                predictedData = np.empty((0,1), float)
        else:
                trueData = np.empty((0,3), float)
                predictedData = np.empty((0,3), float)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		with open(filename) as infile:
			count = 0
			sum_mean_error = 0
			nOutliers = 0
			for line in infile:
				im = np.fromstring(line, dtype=np.float32, sep=',')

				img = np.reshape(im[0:74*350], (1,74*350))

				if trainEnergy:
					labels = np.reshape(im[74*350+3:74*350+4], (1,1))
				else:
					labels = np.reshape(im[74*350:74*350+3], (1,3))

				predicted, mean_dist = sess.run([H_pool, mean_error], feed_dict={x: img, y_: labels})

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

		unit = "mm"
		scale = 1
                if trainEnergy:
                        unit = "%"
			scale = 100
                print("mean distance = %.2f %s"%(sum_mean_error*scale,unit))
		print("mean = %.2f %s"%(np.mean(np.sqrt(np.sum(np.square(predictedData-trueData), axis=1))),unit))
		print("mean x = %.2f %s"%(np.mean(np.absolute(predictedData[:,0]-trueData[:,0])),unit))
		print("mean y = %.2f %s"%(np.mean(np.absolute(predictedData[:,1]-trueData[:,1])),unit))
		print("mean z = %.2f %s"%(np.mean(np.absolute(predictedData[:,2]-trueData[:,2])),unit))

                #title = "mean error = %.2f %s"%(sum_mean_error*scale,unit)
		title = ''

		dsp = APDDisplay()
                dsp.PlotHistos(predictedData, trueData, title, evalPlotName, trainEnergy, filename)
                dsp.DisplayPosition(predictedData, trueData, False, evalPlotName.replace('.','_2D.'), title, scaleLabels)
		#dsp.PlotPositionDiff(predictedData, trueData, evalPlotName.replace('.','_posDiff.'))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
        parser.add_argument('--inputFile', type=str)
        parser.add_argument('--model', type=str)
	parser.add_argument('--trainEnergy', action='store_true')
	parser.add_argument('--evalPlotName', type=str, default='')
        parser.add_argument('--evalFilename', type=str, default='')
	parser.add_argument('--scaleLabels', action='store_true')

	args = parser.parse_args()

	Eval(args.inputFile, args.model, args.evalPlotName, args.trainEnergy, args.scaleLabels)
	
