import sys

'''
This is an unsupervised learning dataset, hence it accepts inputs only
'''
class dataPoint:

	def __init__(self,x,y):
		#x is input, y is output
		self.inputArr = [];
		self.inputArr.append(x);
		self.inputArr.append(y);		

	def getInputArr(self):
		return self.inputArr;	


class dataSet:

	def __init__(self):
		#array of dataPoints
		self.dataArr = [];

	def addPoint(self, point):
		#insert a dataPoint object in array
		self.dataArr.append(point); 

	def getData(self):
		return self.dataArr;
