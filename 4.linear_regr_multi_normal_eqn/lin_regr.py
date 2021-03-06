import sys
import numpy as np


class linearRegrNormalEquation:

	def __init__(self, dataset):
		#dataset is an array of ([],y) objects 
		
		self.input_mat = self.getInputMatrix(dataset);		

		self.output_vec = self.getOutputVector(dataset);
		
		self.theta = []

	def runNormalEquation(self):

		input_transp = np.transpose(self.input_mat)
		#print('Transpose:')
		#print(input_transp)

		product = np.matmul(input_transp, self.input_mat)
		#print('Product:')
		#print(product)

		product_inv = np.linalg.inv( product );
		#print('Product Inverse:')
		#print(product_inv)

		inv_and_transp = np.matmul(product_inv, input_transp)
		#print('Inverse X Transpose:')
		#print(inv_and_transp)

		self.theta = np.matmul(inv_and_transp, self.output_vec)
						

	def doArrRounding(self,arr):
		for val in arr:
			val = round(val,5)

	def getInputMatrix(self, dataset):
		resultArr = []
		for point in dataset:
			resultArr.append(point.getInputArr());
		result = np.array(resultArr);#allows matrix operations
		print('Input Matrix:')
		print(result)
		return result;

	def getOutputVector(self, dataset):
		resultArr = []
		for point in dataset:
			resultArr.append(point.getOutput());
		result = np.array(resultArr);#allows matrix operations
		print('Output Vector:')
		print(result)
		return result;


	def getTheta(self):
		return self.theta;

	
