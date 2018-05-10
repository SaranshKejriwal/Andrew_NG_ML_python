import sys
from dataset_multi import *
from lin_regr import *

def main():
	dataset = createDataset();
	regr_module = linearRegrNormalEquation(dataset.getData(), 2);#dataset, lambda

	regr_module.runNormalEquation();

	print('Normal Equation computed - parameter values:')
	print('Theta:'+str(regr_module.getTheta()))
	
	

def createDataset():
	set1 = dataSet();
	#add dataPoints to set	
	
	set1.addPoint(dataPoint([1,1,1],1))# x0=1 everywhere
	set1.addPoint(dataPoint([1,2,2],2))
	set1.addPoint(dataPoint([1,3.01,3],3))#slight value change to prevent sigularity
	set1.addPoint(dataPoint([1,6,6],6))
	set1.addPoint(dataPoint([1,5,5],5))	
	set1.addPoint(dataPoint([1,4,4],4))	
	#should converge at theta = [1,0.5,0.5]
	
	
	

	return set1;


main()
