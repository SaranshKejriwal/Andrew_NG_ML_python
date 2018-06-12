import sys
import numpy as np
import math
from dataset_multi import *; #used to create points from similarity vectors

class SVM_ModuleMultivariant:

	def __init__(self, alpha, sigma):
		#theta is parameter array for input and theta0 is independent param 
		#alpha is the learning rate

		'''
		Note: theta initialization would have to be removed from constructor, since theta now depends on the number of training examples, not the number of features
		'''
		#self.th = np.array(theta);		

		self.a = alpha;
		self.sigma = sigma;#used in similarity function

	def train(self, datasetObj, iterations):
		self.landmarkSet = datasetObj.getData()
		self.applyGradientDescent(datasetObj.getData(), iterations);

	def applyGradientDescent(self,dataset, iterations):

		#compute 'similarity' of each point with all the other points 

		m = len(dataset);

		self.th = initArrayOfSize(m);		
		#for each dataset point, we create a "similarity feature vector", each of size equal to m
		similaritySet = self.getSimilaritySet(dataset);

		i=0;
		while(i<iterations):
			print('Iteration '+str(i))
			th_err = self.th;			
			
			#compute summation of error
			for point in similaritySet: #not dataset
				#print('point - '+str(point.getInputArr()))
				#theta0_err = theta0_err + self.getCostForDatapoint(point)
				#theta1_err = theta1_err + (point.getInputArr() * self.getCostForDatapoint(point))
				th_err = th_err + (point.getInputArr() * self.getCostForDatapoint(point))

			temp0 = self.th - (self.a * th_err)/m
			#temp1 = self.t1 - (self.a * theta1_err)/m
			temp0 = self.doArrRounding(temp0)
			#temp0 = round(temp0,7)
			#temp1 = round(temp1,7) #prevent highly precise computation
			
			
			if((temp0 == self.th).all()):# .all() method needed for element by element comparison
				print('Parameter calculation complete:')
				break				
			else:
				self.th = temp0				
				print('theta: '+str(self.th));	
			i = i+1;
			#break	

	def getSimilaritySet(self, dataset):
		
		similaritySet = []; #this array will hold the similarity feature vectors for each array.

		for point in dataset:
			simlFeatureVector = [];#this vector holds the similarities of that point with all landmarks
			for landmark in self.landmarkSet:#created a copy of dataset earlier. Needed to create similarity vector for test data
				simlFeatureVector.append(self.computeSimilarity(point.getInputArr(), landmark.getInputArr()));	
			print('Similarity vector for '+str(point.getInputArr())+': '+str(simlFeatureVector));

			tempPoint = dataPoint(simlFeatureVector, point.getOutput());

			similaritySet.append(tempPoint);	
			
		return similaritySet;
	
			
	#this function returns the net distance between 2 vectors - single value
	def computeSimilarity(self, pointInp, landmarkInp):
		numFeatures = len(pointInp);#same as landmarkInp
		
		squaredDistance = 0
		for i in range(numFeatures):
			difference = pointInp[i] - landmarkInp[i];
			squaredDistance = squaredDistance + difference*difference;

		resultExponent = squaredDistance/2*(self.sigma * self.sigma);	
		#print('similarity between '+str(pointInp)+' and '+str(landmarkInp)+' is '+str(math.exp(-1 * resultExponent)));	
		#rounding off very small values
		if resultExponent >400:
			return 0;		
		return math.exp(-1 * resultExponent);#-ive sign accounted for here
		
	'''
	Instead of a sigmoid, the SVM returns 1 if the input is positive		
	def sigmoid(self, x):
		if x < -500:
			return 0; #avoiding math range error
		result = 1/(1 + math.exp(-x))
		return result	
	'''			
	def sqWave(self, x):
		if x>=0:
			return 1;
		else:
			return 0;	

	def doArrRounding(self,arr):
		return np.around(arr,4);#round-off all indices of the array
			


	def getHypothesisForInput(self, input_arr):
		#Note np.dot is the same as trans(theta) * x
		#result = np.dot(self.th,input_arr) #linear regr
		#result = self.sigmoid(np.dot(self.th,input_arr)) #logical regr
		result = self.sqWave(np.dot(self.th,input_arr)) #SVM
		if not np.isnan(result):
			return result
		else:
			return 0;

	def getCostForDatapoint(self, point):
		'''
		y = point.getOutput();
		hx = self.getHypothesisForInput(point.getInputArr())
		#print('y = '+str(y))
		#print('h(x) = '+str(hx))
		result = -1 * (y*math.log(hx) + (1-y)*math.log(1-hx) )
		'''
		#return result
		
		return self.getHypothesisForInput(point.getInputArr()) - point.getOutput();

	def getTheta(self):
		return self.th;


	def test(self, inputArr):
		simlFeatureVector = [];#this vector holds the similarities of that point with all landmarks
		for landmark in self.landmarkSet:			
			simlFeatureVector.append(self.computeSimilarity(inputArr, landmark.getInputArr()));	
		return self.getHypothesisForInput(simlFeatureVector);

		

'''
this method returns an array of zeros of specified size
'''
def initArrayOfSize(m):
	return [0]*m;
		

	
