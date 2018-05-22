import sys
from neuron import *
from neuralNetwork import *
from neuronLayer import *
from dataset_multi import *

def main():
	
	'''
	The objective of this neural network is to use backProp and gradient descent to update a NN design to return XNOR,
	to return XOR instead
	'''

	dataset = createDataset();

	and_nrn = neuron([-30,20,20]);#bias unit comes first
	or_nrn = neuron([-10,20,20]);
	
	nor_nrn = neuron([10,-20,-20]);
	nand_nrn = neuron([30,-20,-20]);

	#XNOR implementation_____________________________________________________

	#Input layer need not be created separately since the input array serves as input layer	
	hidn_lyr_1 = neuronLayer();# this layer has 2 neurons and expects 3 inputs (bias, x1, x2)
	hidn_lyr_1.addNeuron(and_nrn);
	hidn_lyr_1.addNeuron(nor_nrn); #XNOR = (AND) OR (NOR)

	outp_lyr = neuronLayer();
	outp_lyr.addNeuron(or_nrn);

	xnor_nn = neuralNetwork();
	xnor_nn.addNeuronLayer(hidn_lyr_1);
	xnor_nn.addNeuronLayer(outp_lyr);

	print('0 XOR 0: '+str(xnor_nn.getOutputArr([1,0,0])));#bias unit comes first
	print('0 XOR 1: '+str(xnor_nn.getOutputArr([1,0,1])));#bias unit comes first
	print('1 XOR 0: '+str(xnor_nn.getOutputArr([1,1,0])));#bias unit comes first
	print('1 XOR 1: '+str(xnor_nn.getOutputArr([1,1,1])));#bias unit comes first
	print('Note: First element is bias term');

	'''
	XNOR neural network created. Computing error for XOR
	'''

	xnor_nn.computeNetError(dataset);

def createDataset():
	set1 = dataSet();
	#creating dataset for XOR
	
	set1.addPoint(dataPoint([1,0,0],0))
	set1.addPoint(dataPoint([1,0,1],1))
	set1.addPoint(dataPoint([1,1,0],1))
	set1.addPoint(dataPoint([1,1,1],0))# x0=1 everywhere	

	return set1;
	
main();
