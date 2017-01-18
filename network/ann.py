"""
MODULE TO CREATE AND TRAIN AN ARTIFICAL NEURAL NETWORK

The neural network contains an input layer, hidden layer(s) and an output layer.
It may contain several hidden layers with personnalized number of neurons.
The activation function of the hidden neurons and the output neurons can be set.

Date: January 2017
Author: Mareva BRIXY
"""

# coding: utf8
from __future__ import unicode_literals

import math
import pickle
import numpy as np

import activation_function

class Network:

	""" This class defines a neural network structure and train it. """

	def __init__(self, dims):
		""" The constructor of a Network element consists of a list of integers that correspond to
		the number of neurons in each layer.

		eg: [2, 3, 2, 1] is a neural network with an input layer of 2 neurons, 2 hidden layers of 3 and 2 neurons respectively and
		an output layer of 1 neuron.

		Arguments:
		dims                	list           				number of neurons for each layer

		Modified attributes:
		dims 					list 						number of neurons for each layer
		neurons 				list 						neurons values (initialized with 0)
		induced_local_field 	list 						induced local field for each neuron (initialized with 0)
		weights 				list 						weights values between each pair of successive layers (list of matrices)
		bias 					list 						bias values for each layer

		"""

		self.dims = dims
		self.neurons = [np.zeros(d) for d in dims]
		self.induced_local_field = [np.zeros(d) for d in dims]
		self.weights = [np.random.uniform(-1, 1, [dims[i + 1], dims[i]]) for i in range(0, len(dims) - 1)]
		self.bias = [np.random.uniform(-1, 1, d) for d in dims]


	def set_activation_fct(self, activation_fct_h, activation_fct_o):
		""" This function sets the activation function for the hidden layers and
		the output layer. It uses the class ActivationFunction defined in activation_function.py.

		Arguments:
		activation_fct_h    	ActivationFunction object 	activation function for hidden layers
		activation_fct_o    	ActivationFunction object 	activation function for output layer

		Modified attributes:
		activation_fct_h    	ActivationFunction object 	activation function for hidden layers
		activation_fct_o    	ActivationFunction object 	activation function for output layer

		"""

		self.act_fct_h = activation_fct_h
		self.act_fct_o = activation_fct_o


	def feedforward(self, x):
		""" This function enables to feed the whole network with an input vector x.
		Based on this given input vector, the neurons values are computed.

		Arguments:
		x    					vector 						data sample

		"""

		# Feed the network with an input vector x
		self.neurons[0] = x

		# Forward the neurons activation through the network
		for s in range(1, len(self.neurons)):

			# Induced local field
			self.induced_local_field[s] = [np.dot(self.neurons[s - 1], self.weights[s - 1][j])
											for j in range(len(self.neurons[s]))] + self.bias[s]

			# Hidden layers
			if s < len(self.neurons) - 1:
				self.neurons[s] = self.act_fct_h.function(self.induced_local_field[s])
			# Output layer
			else:
				self.neurons[s] = self.act_fct_o.function(self.induced_local_field[s])


	def backpropagate(self, learning_rate, label):
		""" This function enables the update of the weights values thanks to the back propagation
		of the network error from the output layer to the input layer.
		The network error corresponds to the difference between the current output neurons values and the expected label.

		Arguments:
		learning_rate    		double 						learning_rate for the weights update
		label 					array 						expected output, same size as the output layer
		"""

		delta_next_layer = 0

		for s in range(len(self.weights))[::-1]:

			# Output layer
			if s == len(self.weights) - 1:
				network_error = np.array(label - self.neurons[-1])
				activ_fct_term = self.act_fct_o.derivative(self.induced_local_field[-1])
				delta = network_error * activ_fct_term

			# Hidden layers
			else:
				network_error = [np.sum(np.dot(delta_next_layer, self.weights[s + 1][:,j])) for j in range(self.dims[s + 1])]
				activ_fct_term = self.act_fct_h.derivative(self.induced_local_field[s + 1])
				delta = network_error * activ_fct_term

			delta_next_layer = delta

			# Update the weights
			self.weights[s] = self.weights[s] + learning_rate * np.outer(delta, self.neurons[s])
			# Update the bias
			self.bias[s + 1] = self.bias[s + 1] + learning_rate * delta


	def train(self, data, label, learning_rate, nb_epochs):
		""" This functin enables to train a neural network. The network is trained with the dataset 'data' with
		the corresponding expected answers 'label'.
		A sample of the training dataset corresponds to a row of 'data'.
		Let N be the number of samples, P the number of features, C the number of output dimension.

		Arguments:
		data    				N x P array 				dataset
		label 					N x C array  				expected answer / label
		learning_rate    		double 						learning_rate for the weights update
		nb_epochs				integer 					number of epochs (limit)
		"""

		for epoch in range(nb_epochs):

		 	# Sequential learning: sample by sample
		 	for i in range(len(data)):

		 		# Sample
		 		x = data[i,:]
		 		# Feedforward step
		 		self.feedforward(x)
		 		# Backpropagation
		 		self.backpropagate(learning_rate, label[i,:])

		 	print("Epoch {}: OK".format(epoch + 1))


	def test(self, data, label, verb_print=True):
		""" This function enables to test a neural network after training.
		A softmax regression is applied on the output layer and gives therefore only one prediction element.
		Let N be the number of samples and P the number of features.

		Arguments:
		data    				N x P array 				dataset
		label 					N vector  					expected answer / label
		verb_print 				boolean 					information printing

		Returns:
		res 					dictionnary 				contains results of the test phase
			'prediction'		vector 						predictions made by the network
			'accuracy' 			double 						accuracy obtained


		"""

		prediction = np.zeros(len(label))

		for i in range(len(data)):

		 	# Feedforward step
		 	self.feedforward(data[i,:])
	 		# Softmax regression
	 		softmax_vec = np.vectorize(math.exp)
	 		softmax = softmax_vec(self.neurons[-1]) / sum(self.neurons[-1])
			# Classification
			prediction[i] = np.argmax(softmax)

		accuracy = np.mean(prediction == label)

		if verb_print:
			print("Accuracy obtained: {}".format(accuracy))

		res ={}
		res['prediction'] = prediction
		res['acc'] = accuracy

		return res


	def save(self, filename):
		""" This function enables to save the current state of the network into a file.

		Arguments:
		filename 				string 						name of the file that contains the saved network
		"""

		with open(filename, 'wb') as myfile:
    		pickler = pickle.Pickler(myfile)
    		pickler.dump(self)

		print "The network has been saved into the file 'network_pickle.p'. "


	def load(self, filename):
		""" This function enables to load a network.

		Arguments:
		filename 				string 						name of the file that contains the saved network

		Returns:
		network 				Network object 				network as described in this class
		"""

		with open(filename, 'rb') as myfile:
			depickler = pickle.Unpickler(myfile)
			network = depickler.load()

		print "The network has been successfully loaded."

		return network



