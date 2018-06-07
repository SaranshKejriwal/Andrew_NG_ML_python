import sys
import numpy as np
from neuronLayer import * 

#this class corresponds to a single layer of
class neuralNetwork:

	def __init__(self, inputArr):
		#numNeurons is the number of neurons in that layer
		#inputArr is the array of outputs of the previous layer, or the input data at the input layer
		self.inputLayer = neuronLayer(len(inputArr), inputArr);
		self.neuronLayerArr = [];
		self.inputArr = inputArr;
		self.neuronLayerArr.append(self.inputLayer);

	#this constructor only creates a blank array, assuming that neurons will be injected separately	
	def __init__(self):
		self.neuronLayerArr = [];		
		
	def addNeuronLayer(self, neuronLayer):
		self.neuronLayerArr.append(neuronLayer);

	def getNeuronLayerAtIdx(self, idx):
		return self.neuronLayerArr[idx];

	def getNeuronLayerArr(self):
		return self.neuronLayerArr;

	#this method returns the hypothesis for a given input
	def getOutputArr(self, inputArr):
		#return output without bias unit
		outputArr = np.delete(self.propagateForward(inputArr,len(self.neuronLayerArr)-1),0);#remove 0th term
		return np.around(outputArr,3);#round-off all indices of the array
		
		
	#This method to be called only when all layers are appended to the network object
	def propagateForward(self, inputArr, idx):
		#this method uses recursion to supply a layer's output to the next layer. Only Input layer uses input array		
		if idx<=0:
			return self.neuronLayerArr[0].getOutputArr(inputArr);
		else:
			return self.neuronLayerArr[idx].getOutputArr(self.propagateForward(inputArr, idx-1));
			
	def propagateBack(self, expectedPoint):
		networkError = []; #used to hold delta values for each layer
		for i in range (0, len(self.neuronLayerArr)):
			networkError.append(0);
		#separate computation for output layer
		#outputLayer is self.neuronLayerArr[len(self.neuronLayerArr)-1];
		hypothesis = self.getOutputArr(expectedPoint.getInputArr());#this updates the layer params
		lastLayerIdx = len(self.neuronLayerArr)-1# len starts from 1, not 0
		print('__________\nHypothesis for '+str(expectedPoint.getInputArr())+' : '+str(hypothesis));
		print('Expected Output for '+str(expectedPoint.getInputArr())+' : '+str(expectedPoint.getOutput()));

		#networkError[lastLayerIdx] holds output layer error
		networkError[lastLayerIdx] = hypothesis - expectedPoint.getOutput();# these don't include bias terms
		#output layer is currently 1 neuron, for single class classif
		self.neuronLayerArr[lastLayerIdx].addToNetError(networkError[lastLayerIdx]);
		#add net delta from all training examples
		print('Error for '+str(expectedPoint.getInputArr())+' : '+str(networkError[lastLayerIdx]));

		for i in range (len(self.neuronLayerArr)-2 , 1): #loop covers all hidden layers
			#supply networkError on (i+1)th layer to i'th layer
			networkError[i] = self.neuronLayerArr[i].updateBackPropError(networkError[i+1]);
			#updateBackPropError returns the error vector of that layer to the network level matrix, 
			#and adds the error of that training example to net error array of that layer
			
		#print('Neural Network error:\n'+str(networkError));
		return networkError;#this array of errors per neuron isn't used
	'''
	#this method ensures that a positive error and a negative error don't cancel one another out ':)
	def modulus(self,inputNum):
		if(inputNum >=0):
			return inputNum;
		else:
			return -1*inputNum;
	'''

	def resetAllLayerErrors(self):
		#this method resets the net error of all layers, that was added up over the last computeNetError call
		for layer in self.neuronLayerArr:
			layer.resetNetError();

	def updateModel(self, dataset):
		i=0
		m=len(dataset.getData())
		#while True:#risky
		while i<100:		
			for layer in self.neuronLayerArr:
				#print('Avg error of layer '+str(i)+' over all examples: '+str(layer.netError/m));
				layer.updateNeurons(m);
				i=i+1;
			self.computeNetError(dataset);#keep doing backprop, till the error (hopefully) minimizes
		

	def computeNetError(self, dataset):
		self.resetAllLayerErrors();#incremental computations shouldn't stack
		for point in dataset.getData():
			#self.propagateForward(point.getInputArr(), len(self.neuronLayerArr)-1);#this updates the layer outputs
			self.propagateBack(point)#this is the training example
		i=0
		for layer in self.neuronLayerArr:
			print('Net error of layer '+str(i)+' over all examples: '+str(layer.netError));
			i=i+1;
		print('__________________________________________________')
		#warning - reset each layer's net error before calling this method again to prevent stacking

