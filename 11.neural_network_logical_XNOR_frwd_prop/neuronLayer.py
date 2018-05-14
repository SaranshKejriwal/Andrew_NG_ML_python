import sys
import numpy as np
from neuron import *

#this class corresponds to a single layer of
class neuronLayer:

	#this constructor initialises the layer with default neurons
	def __init__(self, numNeurons, numInputs):
		#numNeurons is the number of neurons in that layer
		#numInputs is the number of outputs of the previous layer, or the number of features at input layer
		self.neuronArr = [];
		self.numNeurons = numNeurons;
		for i in range(numNeurons):
			temp_paramArr = self.getInitParamArr(numInputs);
			temp_neuron = neuron(temp_paramArr)
			self.neuronArr.append(temp_neuron); #initialised neuron added to array

	#this constructor only creates a blank array, assuming that neurons will be injected separately
	def __init__(self):
		#numInputs is the number of outputs of the previous layer, or the number of features at input layer
		self.neuronArr = [];
		
	def addNeuron(self,neuron):
		self.neuronArr.append(neuron);

	def getNumNeurons(self):
		return self.numNeurons;

	def getNeuronArr(self):
		return self.neuronArr;

	def getOutputArr(self, inputArray):
		#inputArray is the input to compute layer output for
		#addBias is a boolean that decides whether a bias unit should be appended to the output array
		#Bias is added for all hidden layers, but not output layer
		#print('Input Array:'+str(inputArray));
		resultArr = [];
		for neuron in self.neuronArr:
			resultArr.append(neuron.getOutput(inputArray));
		
		#add bias term 1 at the beginning of the array
		result = np.array(self.getArrWithBiasUnit(resultArr));		
		return result;

	#this method initialises a parameter array for a neuron based on its number of inputs. Not used for now
	def getInitParamArr(self, arrSize):
		temp = []
		for i in range(arrSize):
			temp.append(1); #add 1 element to array for dot product
		result = np.array(temp);
		return result;

	#this method appends a bias unit as the first element of the result
	def getArrWithBiasUnit(self, arr):
		result = []
		result.append(1);#bias unit
		result.extend(arr);#concat
		return result;
		

