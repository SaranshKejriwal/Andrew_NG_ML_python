import sys


class linearRegrModuleUni:

	def __init__(self, theta0, theta1, alpha):
		#theta1 is parameter for input and theta0 is independent param 
		#alpha is the learning rate
		self.t1 = theta1;
		self.t0 = theta0;

		self.a = alpha;
		

	def applyGradientDescent(self,dataset):

		print('t1: '+str(self.t1)+'; t0: '+str(self.t0));
		m = len(dataset);
		
		while(True):
			theta0_err = 0
			theta1_err = 0

			#compute summation of error
			for point in dataset:
				theta0_err = theta0_err + self.getCostForDatapoint(point)
				theta1_err = theta1_err + (point.getInput() * self.getCostForDatapoint(point))

			temp0 = self.t0 - (self.a * theta0_err)/m
			temp1 = self.t1 - (self.a * theta1_err)/m

			temp0 = round(temp0,7)
			temp1 = round(temp1,7) #prevent highly precise computation
			
			
			if((temp0 == self.t0) or (temp1 == self.t1)):
				print('Parameter calculation complete:')
				break				
			else:
				self.t0 = temp0
				self.t1 = temp1
				print('t1: '+str(self.t1)+'; t0: '+str(self.t0));

			

			
						



	def getHypothesisForInput(self, input_val):
		return (input_val*self.t1) +self.t0

	def getCostForDatapoint(self, point):
		return self.getHypothesisForInput(point.getInput()) - point.getOutput();

	def getTheta0(self):
		return self.t0;

	def getTheta1(self):
		return self.t1;
