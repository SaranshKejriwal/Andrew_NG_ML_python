import sys

class dataPoint:

	def __init__(self,x,y):
		#x is input, y is output
		self.input = x;
		self.output = y;

	def getInput(self):
		return self.input;

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
