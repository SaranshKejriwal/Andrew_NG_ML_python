import sys
import numpy as np
import math

class neuron:

	def __init__(self, paramArr):
		#paramArr is the matrix of parameters of a neuron, inputArr is the input vector
		self.paramArr = np.array(paramArr);#allows matrix operations
		self.error = 0;			
	
	def getParamArr(self):
		return self.paramArr;


	def getOutput(self, inputArr):
		if len(inputArr) != len(self.paramArr):
			print('Cannot compute neuron output. Input array and param array should be comparable in size')
		#return round(self.sigmoid(np.dot(inputArr, self.paramArr)),2); #rounding removed for derivative
		self.output = self.sigmoid(np.dot(inputArr, self.paramArr))
		return self.output;

	def sigmoid(self, x):
		if x < -500:
			return 0; #avoiding math range error
		result = 1/(1 + math.exp(-x))
		return result


	def setError(self, error):
		self.error = error;
	
	def getError(self):
		return self.error;

	def getDerivOfSigmoid(self):
		#this returns the derivative of sigmoid a(1-a), where a = g(z)
		return self.output * (1-self.output); 
		
