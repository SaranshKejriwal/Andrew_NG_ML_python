import sys
from neuron import *

def main():
	
	and_nrn = neuron([-30,20,20]);#bias unit comes first
	or_nrn = neuron([-10,20,20]);
	not_nrn = neuron([10,-20]);

	nor_nrn = neuron([10,-20,-20]);
	nand_nrn = neuron([30,-20,-20]);
	print('1 AND 0: '+str(and_nrn.getOutput([1,0,1])))#bias unit comes first
	print('1 AND 1: '+str(and_nrn.getOutput([1,1,1])))#bias unit comes first
	print('0 AND 0: '+str(and_nrn.getOutput([1,0,0])))#bias unit comes first
	print('__________________')
	print('1 OR 0: '+str(or_nrn.getOutput([1,0,1])))#bias unit comes first
	print('0 OR 0: '+str(or_nrn.getOutput([1,0,0])))#bias unit comes first
	print('1 OR 1: '+str(or_nrn.getOutput([1,1,1])))#bias unit comes first
	print('__________________')
	print('NOT 0: '+str(not_nrn.getOutput([1,0])))#bias unit comes first
	print('NOT 1: '+str(not_nrn.getOutput([1,1])))#bias unit comes first
	print('__________________')
	print('1 NOR 0: '+str(nor_nrn.getOutput([1,0,1])))#bias unit comes first
	print('0 NOR 0: '+str(nor_nrn.getOutput([1,0,0])))#bias unit comes first
	print('1 NOR 1: '+str(nor_nrn.getOutput([1,1,1])))#bias unit comes first
	print('__________________')
	print('1 NAND 0: '+str(nand_nrn.getOutput([1,0,1])))#bias unit comes first
	print('1 NAND 1: '+str(nand_nrn.getOutput([1,1,1])))#bias unit comes first
	print('0 NAND 0: '+str(nand_nrn.getOutput([1,0,0])))#bias unit comes first
	print('__________________')
	
	
main()
