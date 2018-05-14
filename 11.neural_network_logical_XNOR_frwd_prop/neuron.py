import sys
import numpy as np
import math

class neuron:

	def __init__(self, paramArr):
		#paramArr is the matrix of parameters of a neuron, inputArr is the input vector
		self.paramArr = np.array(paramArr);#allows matrix operations		
	
	def getParamArr(self):
		return self.paramArr;


	def getOutput(self, inputArr):
		if len(inputArr) != len(self.paramArr):
			print('Cannot compute neuron output. Input array and param array should be comparable in size')
		return round(self.sigmoid(np.dot(inputArr, self.paramArr)),2);

	def sigmoid(self, x):
		if x < -500:
			return 0; #avoiding math range error
		result = 1/(1 + math.exp(-x))
		return result



