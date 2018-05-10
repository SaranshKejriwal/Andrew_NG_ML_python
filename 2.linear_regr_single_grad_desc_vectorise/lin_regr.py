import sys
import numpy as np


class linearRegrModuleMultivariant:

	def __init__(self, theta, alpha):
		#theta is parameter array for input and theta0 is independent param 
		#alpha is the learning rate
		self.th = theta;		

		self.a = alpha;
		

	def applyGradientDescent(self,dataset):

		print('theta: '+str(self.th));
		m = len(dataset);
		
		while(True):
			th_err = np.array([0,0]);#hard coded for 1 feature			
			
			#compute summation of error
			for point in dataset:
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
			#break	

			
						

	def doArrRounding(self,arr):
		return np.around(arr,5);#round-off all indices of the array
		#for val in arr:
			#val = round(val,2)


	def getHypothesisForInput(self, input_arr):
		#Note np.dot is the same as trans(theta) * x
		result = np.dot(self.th,input_arr)
		if not np.isnan(result):
			return result

	def getCostForDatapoint(self, point):
		return self.getHypothesisForInput(point.getInputArr()) - point.getOutput();

	def getTheta(self):
		return self.th;

	
