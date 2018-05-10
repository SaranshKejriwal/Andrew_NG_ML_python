import sys
import cv2
import cv
import numpy as np

class linearRegrModuleUnivariant:

	def __init__(self, theta0, theta1, alpha):
		#theta1 is parameter for input and theta0 is independent param 
		#alpha is the learning rate
		self.t1 = theta1;
		self.t0 = theta0;

		self.a = alpha;

		self.graph = cv2.imread("ui_blank.jpg")
		

	def applyGradientDescent(self,dataset):

		#print('t1: '+str(self.t1)+'; t0: '+str(self.t0));
		m = len(dataset);
		cv2.namedWindow('graph')
		cv2.moveWindow('graph',5,5)

		self.drawPoints(dataset)
		
		while(True):
			self.graph = cv2.imread("ui_blank.jpg")
			
			theta0_err = 0
			theta1_err = 0

			self.computeError(dataset)
			#compute summation of error
			
			self.drawLine()
			self.drawPoints(dataset)#points should overlap the line
			self.writeTheta()
			cv2.imshow('graph',self.graph)

			if(cv2.waitKey(10) & 0xFF == ord('b')):
        			break

	def computeError(self, dataset):
		m = len(dataset);
		theta0_err = 0
		theta1_err = 0

		#compute summation of error
		for point in dataset:
			theta0_err = theta0_err + self.getCostForDatapoint(point)
			theta1_err = theta1_err + (point.getInput() * self.getCostForDatapoint(point))

		temp0 = self.t0 - (self.a * theta0_err)/m
		temp1 = self.t1 - (self.a * theta1_err)/m

		temp0 = round(temp0,5)
		temp1 = round(temp1,5) #prevent highly precise computation
			
			
		if((temp0 == self.t0) or (temp1 == self.t1)):
			pass
			#print('Parameter calculation complete:')							
		else:
			self.t0 = temp0
			self.t1 = temp1
			#print('t1: '+str(self.t1)+'; t0: '+str(self.t0));

		
	def drawPoints(self,dataset):
		for point in dataset:
			cv2.circle(self.graph, (point.getInput()*30,point.getOutput()*30), 5, (10, 50, 200), 4)

	def writeTheta(self):
		theta0 = 'Theta_0: ' + str(self.t0);
		theta1 = 'Theta_1: ' + str(self.t1);
		cv2.putText(self.graph,theta0,(300,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
		cv2.putText(self.graph,theta1,(300,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
						

	def drawLine(self):
		#if( int(self.getHypothesisForInput(0)) > 0 and int(self.getHypothesisForInput(100)) >0):
		cv2.line(self.graph,(0,int(self.getHypothesisForInput(0))),(200,int(self.getHypothesisForInput(200))),(100,0,100),4)
			#cannot draw lines on negative coordinates


	def getHypothesisForInput(self, input_val):
		return (input_val*self.t1) +self.t0

	def getCostForDatapoint(self, point):
		return self.getHypothesisForInput(point.getInput()) - point.getOutput();

	def getTheta0(self):
		return self.t0;

	def getTheta1(self):
		return self.t1;
