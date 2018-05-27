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
		self.outputArr = [];#this array holds the outputs of the individual neurons;used for backprop
		self.numNeurons = numNeurons;
		for i in range(numNeurons):
			temp_paramArr = self.getInitParamArr(numInputs);
			temp_neuron = neuron(temp_paramArr)
			self.neuronArr.append(temp_neuron); #initialised neuron added to array

		self.netError = 0;#this value holds the net error from backprop on all training examples
		self.backPropError = []; #in case of multi class classification, it's a vector

	#this constructor only creates a blank array, assuming that neurons will be injected externally
	def __init__(self):
		#numInputs is the number of outputs of the previous layer, or the number of features at input layer
		self.neuronArr = [];
		self.outputArr = [];#this array holds the outputs of the individual neurons;used for backprop
		self.netError = 0;#this value holds the net error from backprop on all training examples
		self.backPropError = []; #in case of multi class classification, it's a vector
		
	def addNeuron(self,neuron):
		self.neuronArr.append(neuron);

	def getNumNeurons(self):
		return len(self.neuronArr);

	def getNeuronArr(self):
		return self.neuronArr;

	def getOutputArr(self, inputArray):
		#inputArray is the input to compute layer output for
		resultArr = [];
		for neuron in self.neuronArr:
			resultArr.append(neuron.getOutput(inputArray));
		
		self.outputArr = np.array(resultArr);#outputs without bias unit
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

	def getNetError(self):# this returns the error accumalator value vector
		return self.netError;

	def addToNetError(self, errorOfSingleExample): #called for each point in the training set
		self.netError = self.netError + errorOfSingleExample;

	def updateBackPropError(self,nextLayerError):
		
		#implement weighted sum of errorTerms of nextLayerError
		#error vector delta, doesn't contain bias unit

		print('Error vector received:\n'+str(nextLayerError));

		layerParamArray = self.getParamMatrix(); #consolidate param vectors of individual neurons
		print('Param matrix of layer:\n'+str(layerParamArray));

		#length of the error vector is equal to the number of neurons in this layer
		for neuron in self.neuronArr):
			tempErrorHolder = 0; #create a placeholder for neuron		
			for i in range(len(nextLayerError)):
				tempErrorHolder = tempErrorHolder + neuron.getParamArr()[i+1]*nextLayerError[i];
				#i+1 accounts for the bias unit added to paramArr
			neuron.setError(tempErrorHolder);
			self.backPropError.append(tempErrorHolder);

		return 	self.backPropError;
			
		

	#this method combines the param arrays of individual neurons to get layer level matrix
	def getParamMatrix(self):
		result = [];
		for neuron in self.neuronArr:
			result.append(neuron.getParamArr());#append, not extend
		return np.array(result)# get operable matrix

	def getOutputDerivativeVectorFromNeurons(self):
		result = [];
		for neuron in self.neuronArr:
			result.append(neuron.getDerivOfSigmoid());
		return np.array(result)# get operable matrix

	def getOutputDerivativeVectorFromLayerOutput(self):
		return self.outputArr * (1-self.outputArr);

