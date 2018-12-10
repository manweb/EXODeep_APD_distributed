import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv

class DataSet(object):
	def __init__(self,dataFiles,num_epochs=None,hasWaveforms=False,trainEnergy=False,isOldFile=False):
		self.filenames = dataFiles
		self.numEpochs = num_epochs
		self.hasWaveforms = hasWaveforms
		self.trainEnergy = trainEnergy
		self.isOldFile = isOldFile

	def GetSample(self):
		filename_queue = tf.train.string_input_producer(self.filenames, num_epochs=self.numEpochs, shuffle=True)

		reader = tf.TextLineReader()
		key, value = reader.read(filename_queue)

		nFeatures = 74
		nLabels = 4
		if self.hasWaveforms:
			nFeatures = 74*350

		nColumns = nFeatures+nLabels+1
		if self.isOldFile:
			nColumns += -1
		record_defaults = [[0.0]]*(nColumns)
		row = tf.decode_csv(value, record_defaults=record_defaults)

		features = row[0:nFeatures]
		if self.trainEnergy:
			labels = row[nFeatures+nLabels-1:nFeatures+nLabels]
		else:
			labels = row[nFeatures:nFeatures+nLabels-1]

		return features, labels

	def GetBatch(self, batch_size):
		features, labels = self.GetSample()

		min_after_dequeue = 10000
		capacity = min_after_dequeue + 3*batch_size

		batch_features, batch_labels = tf.train.shuffle_batch([features, labels], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

		return batch_features, batch_labels

	def GetCVSet(self, filename, nsamples, shuffle = False):
		nFeatures = 74
		nLabels = 4
		if self.hasWaveforms:
			nFeatures = 74*350

		features = []
		labels = []
		count = 0
		with open(filename) as infile:
			for line in infile:
				x = np.fromstring(line, dtype=float, sep=',')
				features.append(x[0:nFeatures])

				if self.trainEnergy:
					labels.append(x[nFeatures+nLabels-1:nFeatures+nLabels])
				else:
					labels.append(x[nFeatures:nFeatures+nLabels-1])

				count += 1

				if count > nsamples:
					break

		return np.asarray(features), np.asarray(labels)

