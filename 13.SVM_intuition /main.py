import sys
from dataset_multi import *
from logis_regr_SVM import *

def main():
	dataset = createDataset();
	regr_module = SVM_ModuleMultivariant([0,0,0],50);#[theta0, theta1, theta2], alpha

	regr_module.applyGradientDescent(dataset.getData());

	print('Gradient Descent completed - Optimum parameter values:')
	print('Theta:'+str(regr_module.getTheta()))
	
	print('[2.5,2.5] : '+str(regr_module.getHypothesisForInput([1,2.5,2.5])))	
	print('[4.5,4.5] : '+str(regr_module.getHypothesisForInput([1,4.5,4.5])))
	print('[4.51,4.51] : '+str(regr_module.getHypothesisForInput([1,4.51,4.51])))
	#margin = 0.5, which is max
	print('[4.75,4.75] : '+str(regr_module.getHypothesisForInput([1,4.75,4.75])))
	print('[5,5] : '+str(regr_module.getHypothesisForInput([1,5,5])))
	print('[5.5,5.5] : '+str(regr_module.getHypothesisForInput([1,5.5,5.5])))
	print('[6.5,6.5] : '+str(regr_module.getHypothesisForInput([1,6.5,6.5])))

def createDataset():
	set1 = dataSet();
	#add dataPoints to set	
	
	set1.addPoint(dataPoint([1,1,1],0))# x0=1 everywhere
	set1.addPoint(dataPoint([1,2,2],0))
	set1.addPoint(dataPoint([1,3,3],0))
	set1.addPoint(dataPoint([1,4,4],0))
	set1.addPoint(dataPoint([1,5,5],1))#2-class classification	
	set1.addPoint(dataPoint([1,6,6],1))	
	
	
	

	return set1;


main()
