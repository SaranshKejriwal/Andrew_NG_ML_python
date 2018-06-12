import sys
from dataset import *
from opencv_kmeans import *
import random #to make clusters

def main():
	dataset = createDataset();
	kmeans_module = opencv_kMeansModule(2);#numClusters

	kmeans_module.run_kMeans(dataset.getData());

	kmeans_module.printCentroids();
	

def createDataset():
	set1 = dataSet();	
	
	#make cluster 1
	i=0
	while i<10:
		tempPoint = dataPoint(random.randint(1,40),random.randint(1,50));
		set1.addPoint(tempPoint);
		i=i+1;
	
	#make cluster 2
	i=0
	while i<10:
		tempPoint = dataPoint(random.randint(61,100),random.randint(51,100));
		set1.addPoint(tempPoint);
		i=i+1;

	'''
	#make cluster 3
	i=0
	while i<10:
		tempPoint = dataPoint(random.randint(61,100),random.randint(1,40));
		set1.addPoint(tempPoint);
		i=i+1;
	'''
	
	print('Dataset created');
	return set1;


main()
