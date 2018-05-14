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

	def getOutputArr(self, inputArr):
		return self.propagateForward(inputArr,len(self.neuronLayerArr)-1);
		
		
	#This method to be called only when all layers are appended to the network object
	def propagateForward(self, inputArr, idx):
		#this method uses recursion to supply a layer's output to the next layer. Only Input layer uses input array		
		if idx<=0:
			#addBias is true for input layer
			return self.neuronLayerArr[0].getOutputArr(inputArr);
		else:
			return self.neuronLayerArr[idx].getOutputArr(self.propagateForward(inputArr, idx-1));
			
			
