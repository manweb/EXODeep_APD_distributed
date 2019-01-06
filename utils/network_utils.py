import numpy as np
import tensorflow as tf
from utils.architecture import *

# Get weight variable
def get_weight_variable(name, shape, reader = None):
	if reader:
		return tf.constant(reader.get_tensor(name))
	else:
		return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

# Get bias variable
def get_bias_variable(name, shape, reader = None):
	if reader:
		return tf.constant(reader.get_tensor(name))
	else:
		return tf.get_variable(name, initializer=tf.constant(0.1, shape=shape))

def conv(x, layer, idx, print_info = True, reader = None):
	if print_info: print('\t filter: [%s]'%','.join(map(str, layer['filter'])))
	if print_info: print('\t stride: [%s]'%','.join(map(str, layer['stride'])))
	return tf.nn.conv2d(x, get_weight_variable('W%i_%i'%(layer['id'], idx), layer['filter'], reader), strides=layer['stride'], padding='SAME') + get_bias_variable('b%i_%i'%(layer['id'], idx), [layer['filter'][-1]], reader)

def max_pool(x, layer, print_info = True):
	if print_info: print('\t size: [%s]'%','.join(map(str, layer['filter'])))
	s = layer['filter']
	return tf.nn.max_pool(x, ksize=[1,s[0],s[1],1], strides=[1,s[0],s[1],1], padding='SAME')

def build_network(x, flags, print_info = True, checkpoint_file = None):
	current_layer = x

	reader = None
	if checkpoint_file:
		reader = tf.train.NewCheckpointReader(checkpoint_file)

	for layer in net:
		if layer['type'] == 'conv':
			for idx in range(layer['repeats']):
				if print_info: print('Layer conv%i_%i:'%(layer['id'], idx))
				current_layer = conv(current_layer, layer, idx, print_info, reader)
				if layer['activation'] == 'relu':
					if print_info: print('\t activation: relu')
					current_layer = tf.nn.relu(current_layer)
		elif layer['type'] == 'pooling':
			if print_info: print('Layer maxpool%i'%layer['id'])
			current_layer = max_pool(current_layer, layer, print_info)
		elif layer['type'] == 'reshape':
			if print_info: print('Layer reshape%i'%layer['id'])
			current_layer = tf.reshape(current_layer, layer['filter'])
		elif layer['type'] == 'dropout':
			if print_info: print('Layer dropout%i'%layer['id'])
			keep_prob = tf.constant(flags.dropout, tf.float32)
			current_layer = tf.nn.dropout(current_layer, keep_prob)

	nOutputs = 3
	if flags.trainEnergy:
		nOutputs = 1

	current_layer = tf.nn.conv2d(current_layer, get_weight_variable('W%i_0'%(layer['id']+1), [1,1,layer['filter'][-1],nOutputs], reader), strides=[1,1,1,1], padding='SAME') + get_bias_variable('b%i_0'%(layer['id']+1), [nOutputs], reader)

	current_layer = tf.reshape(current_layer, [-1, nOutputs])

	var = tf.trainable_variables()
	reg = tf.add_n([tf.nn.l2_loss(v) for v in var if 'W' in v.name]) if not checkpoint_file else None

	return current_layer, reg

