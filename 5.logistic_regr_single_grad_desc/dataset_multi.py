import sys
import numpy as np

class dataPoint:

	def __init__(self,x,y):
		#x is input array, y is output
		self.input = np.array(x);#allows matrix operations
		self.output = y;

	def getInputArr(self):
		return self.input;# input is an array

	def getOutput(self):
		return self.output;


class dataSet:

	def __init__(self):
		#array of dataPoints
		self.dataArr = [];

	def addPoint(self, point):
		#insert a dataPoint object in array
		self.dataArr.append(point); 

	def getData(self):
		return self.dataArr;
