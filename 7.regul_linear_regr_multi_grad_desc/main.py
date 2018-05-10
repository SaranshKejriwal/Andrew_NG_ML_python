import sys
from dataset_multi import *
from lin_regr import *

def main():
	dataset = createDataset();
	regr_module = linearRegrModuleMultivariant([1,2,2],0.01,4,True);#[theta0, theta1, theta2], alpha, theta, isRegularized

	regr_module.applyGradientDescent(dataset.getData());

	print('Gradient Descent completed - Optimum parameter values:')
	print('Theta:'+str(regr_module.getTheta()))
	
	

def createDataset():
	set1 = dataSet();
	#add dataPoints to set	
	
	set1.addPoint(dataPoint([1,1,1],1))# x0=1 everywhere
	set1.addPoint(dataPoint([1,2,2],2))
	set1.addPoint(dataPoint([1,3,3],3))
	#set1.addPoint(dataPoint(4,5))
	set1.addPoint(dataPoint([1,5,5],5))	
	set1.addPoint(dataPoint([1,6,6],6))	
	#should converge at theta = [1,0.5,0.5]
	
	
	

	return set1;


main()
