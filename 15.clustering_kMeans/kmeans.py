import sys
import numpy as np
from dataset import *
import random
import math

class kMeansModule:

	def __init__(self, numClusters):
		
		self.centroidArr = []

		i=0;
		#create random centroids `numClusters` times
		while i<numClusters:
			tempCentroid = dataPoint(random.randint(25,75),random.randint(25,75));
			self.centroidArr.append(tempCentroid);
			i=i+1;
		print('Centroids initialized');
		self.printCentroids();

	def run_kMeans(self,dataset, iterations):	
		
		i=0
		
		while(i < iterations):
			print('Iteration '+str(i+1));
			clusterArr = []; #this array will hold the individual clusters of points, which in-turn are also arrays	
			
			#assign dataset points to individual clusters
			clusterArr = self.assignCluster(dataset);
			self.moveCentroids(clusterArr);	
			self.printCentroids();
			i=i+1;
			

	def assignCluster(self, dataset):
		clusterArr = [];

		for i in range(len(self.centroidArr)):
			
			centroid = self.centroidArr[i];
			'''
			The index i is needed in this context because getClosestCentroid output will be compared to i
			'''

			cluster = []; #clusterArr index is parallel to centroidArr index
			for point in dataset:
				if self.getClosestCentroid(point) == i:
					cluster.append(point);	
			clusterArr.append(cluster);

		return clusterArr;

	def moveCentroids(self, clusterArr):
		i=0; #this serves as an index to both clusterArr and centroidArr

		for i in range(len(self.centroidArr)):
			
			cluster = clusterArr[i];# this is the array of points closest to centroid i.
			clusterSize = len(cluster); #number of points
			if(clusterSize == 0):
				print('No points assigned to cluster '+str(i)+'. Ignoring this cluster');				
				continue;
			sum_x = 0;#hold the sum of x-axis values of all points	
			sum_y = 0;#hold the sum of y-axis values of all points	
			for point in cluster:
				sum_x = sum_x + point.inputArr[0];
				sum_y = sum_y + point.inputArr[1];

			avgPoint = dataPoint(sum_x/clusterSize,sum_y/clusterSize);
			self.centroidArr[i] = avgPoint;	
			
	#given a point, this method returns the index of the closest centroid in centroidArr
	def getClosestCentroid(self, point):
		leastDistance = 100000000; #very large value
		i=-1;#this will be incremented at least once by the first point
		for centroid in self.centroidArr:
			centroidDistance = self.getDistanceBetweenPts(centroid, point);
			if(centroidDistance < leastDistance):
				leastDistance = centroidDistance;
				i=i+1;
		#print('Centroid closest to point '+str(point.inputArr)+' : '+str(i));
		return i;


	#this applies distance formula on 2D points only
	def getDistanceBetweenPts(self, p1, p2):
		squaredDistance = 0;
		for i in range(2):
			difference = p1.inputArr[i] - p2.inputArr[i];
			squaredDistance = squaredDistance + difference*difference;
		return math.sqrt(squaredDistance);	

	
	def printCentroids(self):
		print('Centroids computed:');
		for c in self.centroidArr:
			print(c.inputArr);	

