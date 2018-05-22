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
		return np.delete(self.propagateForward(inputArr,len(self.neuronLayerArr)-1),0);#remove first term
		
		
	#This method to be called only when all layers are appended to the network object
	def propagateForward(self, inputArr, idx):
		#this method uses recursion to supply a layer's output to the next layer. Only Input layer uses input array		
		if idx<=0:
			#addBias is true for input layer
			return self.neuronLayerArr[0].getOutputArr(inputArr);
		else:
			return self.neuronLayerArr[idx].getOutputArr(self.propagateForward(inputArr, idx-1));
			
	def propagateBack(self, expectedPoint):
		outputLayerError = []; #used to hold delta values for each layer
		for i in range (0, len(self.neuronLayerArr)):
			outputLayerError.append(0);
		#separate computation for output layer
		#outputLayer is self.neuronLayerArr[len(self.neuronLayerArr)-1];
		hypothesis = self.getOutputArr(expectedPoint.getInputArr());#this updates the layer params
		outputLayerError[len(self.neuronLayerArr)-1] = hypothesis - expectedPoint.getOutput();# these include bias terms
		#output layer is currently 1 neuron, for single class classif
		self.neuronLayerArr[len(self.neuronLayerArr)-1].addToNetError(np.dot(outputLayerError[len(self.neuronLayerArr)-1], expectedPoint.getOutput()));
		print('Error for '+str(expectedPoint.getInputArr())+' : '+str(outputLayerError[len(self.neuronLayerArr)-1]));
		for i in range (len(self.neuronLayerArr)-2 , 1): #loop covers all hidden layers
			#supply outputLayerError on (i+1)th layer to i'th layer
			self.neuronLayerArr[i].updateBackPropError(outputLayerError[i+1]);
		return 0;
		

	def updateModel(self, dataset):
		pass

	def computeNetError(self, dataset):
		for point in dataset.getData():
			#self.propagateForward(point.getInputArr(), len(self.neuronLayerArr)-1);#this updates the layer outputs
			self.propagateBack(point)#this is the training example
		pass
