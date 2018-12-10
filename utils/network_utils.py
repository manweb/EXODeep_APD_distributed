import numpy as np
import tensorflow as tf
from utils.architecture import *

# Get weight variable
def weight_variable(name, shape):
	var = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
	return var

# Get bias variable
def bias_variable(name, shape):
	var = tf.get_variable(name, initializer=tf.constant(0.1, shape=shape))
	return var

def conv(x, layer, idx):
	print('\t filter: [%s]'%','.join(map(str, layer['filter'])))
	print('\t stride: [%s]'%','.join(map(str, layer['stride'])))
	return tf.nn.conv2d(x, weight_variable('W%i_%i'%(layer['id'], idx), layer['filter']), strides=layer['stride'], padding='SAME') + bias_variable('b%i_%i'%(layer['id'], idx), [layer['filter'][-1]])

def max_pool(x, layer):
	print('\t size: [%s]'%','.join(map(str, layer['filter'])))
	s = layer['filter']
	return tf.nn.max_pool(x, ksize=[1,s[0],s[1],1], strides=[1,s[0],s[1],1], padding='SAME')

def build_network(x, flags):
	current_layer = x

	for layer in net:
		if layer['type'] == 'conv':
			for idx in range(layer['repeats']):
				print('Layer conv%i_%i:'%(layer['id'], idx))
				current_layer = conv(current_layer, layer, idx)
				if layer['activation'] == 'relu':
					print('\t activation: relu')
					current_layer = tf.nn.relu(current_layer)
		elif layer['type'] == 'pooling':
			print('Layer maxpool%i'%layer['id'])
			current_layer = max_pool(current_layer, layer)
		elif layer['type'] == 'reshape':
			print('Layer reshape%i'%layer['id'])
			current_layer = tf.reshape(current_layer, layer['filter'])
		elif layer['type'] == 'dropout':
			print('Layer dropout%i'%layer['id'])
			keep_prob = tf.constant(flags.dropout, tf.float32)
			current_layer = tf.nn.dropout(current_layer, keep_prob)

	nOutputs = 3
	if flags.trainEnergy:
		nOutputs = 1

	current_layer = tf.nn.conv2d(current_layer, weight_variable('W%i_0'%(layer['id']+1), [1,1,layer['filter'][-1],nOutputs]), strides=[1,1,1,1], padding='SAME') + bias_variable('b%i_0'%(layer['id']+1), [nOutputs])

	current_layer = tf.reshape(current_layer, [-1, nOutputs])

	var = tf.trainable_variables()
	reg = tf.add_n([tf.nn.l2_loss(v) for v in var if 'W' in v.name])

	return current_layer, reg

