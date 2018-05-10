import sys
from dataset import *
from opencv_lin_regr import *

def main():
	dataset = createDataset();
	regr_module = linearRegrModuleUnivariant(2,2,0.1);#theta0, theta1, alpha

	regr_module.applyGradientDescent(dataset.getData());

	print('Gradient Descent completed - Optimum parameter values:')
	print('Theta_0:'+str(regr_module.getTheta0()))
	print('Theta_1:'+str(regr_module.getTheta1()))
	

def createDataset():
	set1 = dataSet();
	#add dataPoints to set	
	
	set1.addPoint(dataPoint(1,1))
	set1.addPoint(dataPoint(2,2))
	set1.addPoint(dataPoint(3,3))
	#set1.addPoint(dataPoint(4,5))
	set1.addPoint(dataPoint(5,5))	
	set1.addPoint(dataPoint(6,6))	
	
	
	

	return set1;


main()
