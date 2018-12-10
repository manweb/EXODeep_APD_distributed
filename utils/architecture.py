# Definition of the neural network architecture
#
# Import this file when building the graph

net = [	{	'id': 0,
		'type': 'conv',
		'filter': [5,5,1,16],
		'stride': [1,1,1,1],
		'activation': 'relu',
		'repeats': 1
	},
	{	'id': 1,
		'type': 'pooling',
		'filter': [2,3],
		'stride': [2,3]
	},
	{	'id': 2,
		'type': 'conv',
		'filter': [5,5,16,32],
		'stride': [1,1,1,1],
		'activation': 'relu',
		'repeats': 1
	},
	{	'id': 3,
		'type': 'pooling',
		'filter': [2,3],
		'stride': [2,3]
	},
	{	'id': 4,
		'type': 'conv',
		'filter': [5,5,32,64],
		'stride': [1,1,1,1],
		'activation': 'relu',
		'repeats': 1
	},
	{	'id': 5,
		'type': 'pooling',
		'filter': [2,4],
		'stride': [2,4]
	},
	{	'id': 6,
		'type': 'conv',
		'filter': [5,5,64,128],
		'stride': [1,1,1,1],
		'activation': 'relu',
		'repeats': 1
	},
	{	'id': 7,
		'type': 'pooling',
		'filter': [3,3],
		'stride': [3,3]
	},
	{	'id': 8,
		'type': 'reshape',
		'filter': [-1, 1, 1, 2048],
	},
	{	'id': 9,
		'type': 'conv',
		'filter': [1,1,2048,1024],
		'stride': [1,1,1,1],
		'activation': 'relu',
		'repeats': 1
	},
	{	'id': 10,
		'type': 'dropout',
	},
	{	'id': 11,
		'type': 'conv',
		'filter': [1,1,1024,256],
		'stride': [1,1,1,1],
		'activation': 'relu',
		'repeats': 1
	}]

