import sys
import numpy as np
import math

class logisticRegrModuleMultivariant:

	def __init__(self, theta, alpha, lambd, isRegularized):
		#theta is parameter array for input and theta0 is independent param 
		#alpha is the learning rate
		self.th = np.array(theta);		

		self.a = alpha;

		self.lambd = lambd
		self.isRegularized = isRegularized
		

	def applyGradientDescent(self,dataset):

		print('theta: '+str(self.th));
		m = len(dataset);
		
		while(True):
			th_err = np.array([0,0,0]);#hard coded for 2 features			
			
			#compute summation of error
			for point in dataset:
				#theta0_err = theta0_err + self.getCostForDatapoint(point)
				#theta1_err = theta1_err + (point.getInputArr() * self.getCostForDatapoint(point))
				th_err = th_err + (point.getInputArr() * self.getCostForDatapoint(point))

			temp0 = (self.th * self.getRegulznParam(m)) - (self.a * th_err)/m
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
			#break	

			
	def sigmoid(self, x):
		if x < -500:
			return 0; #avoiding math range error
		result = 1/(1 + math.exp(-x))
		return result					

	def doArrRounding(self,arr):
		return np.around(arr,5);#round-off all indices of the array
			


	def getHypothesisForInput(self, input_arr):
		#Note np.dot is the same as trans(theta) * x
		#result = np.dot(self.th,input_arr)
		result = self.sigmoid(np.dot(self.th,input_arr))
		if not np.isnan(result):
			return result

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

	def getRegulznParam(self,m):
		if self.isRegularized:
			return (1 - self.a*self.lambd/m);
		return 1;
	
