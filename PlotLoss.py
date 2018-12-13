import argparse
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import pickle as pk
import glob
import re
import os

def PlotPickle(filenames):
	if len(filenames) > 1:
		print('Multiple pickle files not supported')

		return

	inputFile = filenames[0]

	ax = pk.load(file(inputFile))

	plt.show()

def PlotCSV(filenames, offset = 0, smoothing = False):
	if len(filenames) == 1:
		PlotSingleCSV(filenames[0], offset, smoothing)

		return

	fig = plt.figure(figsize=(12,6))
	ax1 = fig.add_subplot(2,2,1)
	ax2 = fig.add_subplot(2,2,2)
	ax3 = fig.add_subplot(2,2,3)
	ax4 = fig.add_subplot(2,2,4)

	cmap = mpl.cm.viridis
	nFiles = len(filenames)
	lines = []
	for i, inputFile in enumerate(filenames):
		plotx = []
		ploty_train = []
		ploty_cv = []
		ploty_train_cv = []

		matchObj = re.findall(r'\d\.\d+', inputFile)
		regTerm = float(matchObj[0])
		lr = float(matchObj[1])
		if matchObj:
			print('regTerm = %f, learningRate = %f'%(float(matchObj[0]),float(matchObj[1])))
		else:
			print('Nothing found')

		offset2 = 0
        	prevStep = 0	
		with open(inputFile) as infile:
			for line in infile:
				x = np.fromstring(line, dtype=float, sep=',')

				if x[0] < prevStep:
                                	offset2 += prevStep

				plotx.append(x[0]+offset2)
				ploty_train.append(x[1])
				ploty_cv.append(x[2])
				ploty_train_cv.append(np.absolute((x[1]-x[2])/x[1]))

				prevStep = x[0]

		if smoothing:
			ploty_train = smooth(ploty_train)
			ploty_cv = smooth(ploty_cv)
			ploty_train_cv = smooth(ploty_train_cv)
	
		line, = ax1.plot(plotx, ploty_train, color=cmap(i / float(nFiles)), label='rt = %.3fe-3, lr = %.3fe-3'%(regTerm*1000, lr*1000))
		ax2.plot(plotx, ploty_cv, color=cmap(i / float(nFiles)))
		ax3.plot(plotx, ploty_train_cv, color=cmap(i / float(nFiles)))
		lines.append(line)

	ax1.set_title('train loss')
	ax2.set_title('validation loss')

	ax1.set_xlabel('training step')
	ax1.set_ylabel('loss')
	ax2.set_xlabel('training step')
	ax2.set_ylabel('loss')

	ax4.legend(handles=lines, bbox_to_anchor=(0,1.0), loc='upper left', ncol=3, fontsize=6)
	ax4.axis('off')

	plt.show()

def PlotSortedCSV(inputDir):
	fig = plt.figure(figsize=(12,6))
	ax1 = fig.add_subplot(2,2,1)
	ax2 = fig.add_subplot(2,2,2)
	ax3 = fig.add_subplot(2,2,3)
	ax4 = fig.add_subplot(2,2,4)

	cmap = mpl.cm.viridis
	nFiles = 36
	lines = []

	rt = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
	lr = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 4e-4]

	baseFileName = inputDir+'loss_%.7f_%.7f.csv'

	count = 0
	for i in lr:
		for k in rt:
			plotx = []
			ploty_train = []
			ploty_cv = []
			ploty_train_cv = []

			inputFile = baseFileName%(k,i)
			if not os.path.isfile(inputFile):
				print('File %s does not exist'%inputFile)

				continue

			offset2 = 0
        		prevStep = 0		
			with open(inputFile) as infile:
				for line in infile:
					x = np.fromstring(line, dtype=float, sep=',')

					if x[0] < prevStep:
                                		offset2 += prevStep

					plotx.append(x[0]+offset2)
					ploty_train.append(x[1])
					ploty_cv.append(x[2])
					ploty_train_cv.append(np.absolute((x[1]-x[2])/x[1]))

					prevStep = x[0]
		
			line, = ax1.plot(plotx, ploty_train, color=cmap(count / float(nFiles)), label='rt = %.3fe-3, lr = %.3fe-3'%(k*1000, i*1000))
			ax2.plot(plotx, ploty_cv, color=cmap(count / float(nFiles)))
			ax3.plot(plotx, ploty_train_cv, color=cmap(count / float(nFiles)))
			lines.append(line)

			count += 1

	ax1.set_title('train loss')
	ax2.set_title('validation loss')

	ax1.set_xlabel('training step')
	ax1.set_ylabel('loss')
	ax2.set_xlabel('training step')
	ax2.set_ylabel('loss')

	ax4.legend(handles=lines, bbox_to_anchor=(0,1.0), loc='upper left', ncol=3, fontsize=6)
	ax4.axis('off')

	plt.show()

def PlotSingleCSV(inputFile, nEvents, smoothing):
	plotx = []
        ploty_train = []
        ploty_cv = []

	batchSize = 256.0
	epochNum = nEvents/batchSize
	offset2 = 0
	prevStep = 0
	count = 0
	with open(inputFile) as infile:
		for line in infile:
			x = np.fromstring(line, dtype=float, sep=',')

			if x[0]/epochNum < prevStep:
				nRemove = int(floor(len(plotx)/100.0)*100)

				plotx = plotx[:nRemove]
				ploty_train = ploty_train[:nRemove]
				ploty_cv = ploty_cv[:nRemove]

				offset2 = plotx[-1] + 10.0/epochNum

                        plotx.append(x[0]/epochNum+offset2)
                        ploty_train.append(x[2])
                        ploty_cv.append(x[3])

			count += 10

			prevStep = x[0]/epochNum

	fig = plt.figure(figsize=(8,5))

	#plotx, ploty_train, ploty_cv = GetRate(plotx, ploty_train, ploty_cv)

	if smoothing:
		ploty_train = smooth(ploty_train)
		ploty_cv = smooth(ploty_cv)

	plt.plot(plotx, ploty_train, color='blue', label='Training set')
	plt.plot(plotx, ploty_cv, color='green', label='Validation set')

	plt.legend(bbox_to_anchor=(0.58,0.73), fontsize=15, loc='lower left')

	plt.xlabel('Training time [epochs]', fontsize=14)
	plt.ylabel('Loss [mm$^{2}$]', fontsize=14)

	plt.grid(linestyle=':', which='both')

	plt.show()

def GetRate(plotx, ploty_train, ploty_cv):
	r_plotx = []
	r_ploty_train = []
	r_ploty_cv = []

	for i, x in enumerate(plotx):
		if i == 0 or i == len(plotx)-1:
			continue

		r_plotx.append(x)
		r_ploty_train.append(np.absolute(ploty_train[i+1]-ploty_train[i-1])/(plotx[i+1]-plotx[i-1]))
		r_ploty_cv.append(np.absolute(ploty_cv[i+1]-ploty_cv[i-1])/(plotx[i+1]-plotx[i-1]))

	return r_plotx, r_ploty_train, r_ploty_cv

def smooth(x, w=30):
	s = np.r_[2*x[0]-x[w-1::-1],x,2*x[-1]-x[-1:-w:-1]]
	k = np.ones(w,'d')
	y = np.convolve(k/k.sum(),s,mode='same')
	
	return y[w:-w+1]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputFile', type=str, default='')
	parser.add_argument('--nEvents', type=int, default=256)
	parser.add_argument('--smoothing', action='store_true')

	args = parser.parse_args()

	filenames = glob.glob(args.inputFile)

	if filenames[0].endswith('.pickle'):
		PlotPickle(filenames)
	elif filenames[0].endswith('.csv'):
		PlotCSV(filenames, args.nEvents, args.smoothing)
	elif os.path.isdir(filenames[0]):
		PlotSortedCSV(filenames[0])
	else:
		print('Input file type not supported')

