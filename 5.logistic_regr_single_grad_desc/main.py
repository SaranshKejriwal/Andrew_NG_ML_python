import sys
from dataset_multi import *
from logis_regr import *

def main():
	dataset = createDataset();
	regr_module = logisticRegrModuleMultivariant([0,0],50);#[theta0, theta1, theta2], alpha

	regr_module.applyGradientDescent(dataset.getData());

	print('Gradient Descent completed - Optimum parameter values:')
	print('Theta:'+str(regr_module.getTheta()))
	
	

def createDataset():
	set1 = dataSet();
	#add dataPoints to set	
	
	set1.addPoint(dataPoint([1,1],0))# x0=1 everywhere
	set1.addPoint(dataPoint([1,2],0))
	set1.addPoint(dataPoint([1,3],0))
	set1.addPoint(dataPoint([1,4],0))
	set1.addPoint(dataPoint([1,5],1))#2-class classification	
	set1.addPoint(dataPoint([1,6],1))	
	
	
	

	return set1;


main()
