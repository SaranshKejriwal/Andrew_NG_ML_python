import sys
import cv2
import cv
import numpy as np
from dataset import *
import random
import math


scale_multiplier = 3;
#this global value enlarges the point values, to spread them better for drawing on the Mat

class opencv_kMeansModule:

	def __init__(self, numClusters):
		
		self.graph = cv2.imread("ui_blank.jpg")		
		self.centroidArr = []

		i=0;
		#create random centroids `numClusters` times
		while i<numClusters:
			tempCentroid = dataPoint(random.randint(25,75),random.randint(25,75));
			self.centroidArr.append(tempCentroid);
			i=i+1;
		print('Centroids initialized')

	def run_kMeans(self,dataset):
		
		
		cv2.namedWindow('graph')
		cv2.moveWindow('graph',5,5)

		self.drawPoints(dataset)
		self.drawCentroids();
		
		
		while(True):
			self.graph = cv2.imread("ui_blank.jpg")		

			clusterArr = []; #this array will hold the individual clusters of points, which in-turn are also arrays	
			
			#assign dataset points to individual clusters
			clusterArr = self.assignCluster(dataset);
			self.moveCentroids(clusterArr);	
			
			
			self.drawCentroids();
			self.drawDecisionBoundary();
			self.drawPoints(dataset)#points should overlap the line
			self.writeCentroidCoord()
			cv2.imshow('graph',self.graph)

			if(cv2.waitKey(10) & 0xFF == ord('b')):
        			break

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
		
	def drawPoints(self,dataset):
		for point in dataset:
			cv2.circle(self.graph, (point.inputArr[0]*scale_multiplier,point.inputArr[1]*scale_multiplier), 3, (10, 50, 200), 6)

	def drawCentroids(self):
		for point in self.centroidArr:
			cv2.circle(self.graph, (point.inputArr[0]*scale_multiplier,point.inputArr[1]*scale_multiplier), 6, (200, 50, 0), 5)

	def writeCentroidCoord(self):
		for i in range(len(self.centroidArr)):			
			toWrite = 'Centroid_'+str(i+1)+': ' + str(self.centroidArr[i].getInputArr());
			cv2.putText(self.graph,toWrite,(300,(80+32*i)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)	
						

	def connectCentroids(self):
		for point in self.centroidArr:
			for ref in self.centroidArr:
				cv2.line(self.graph,(point.inputArr[0]*scale_multiplier,point.inputArr[1]*scale_multiplier),(ref.inputArr[0]*scale_multiplier,ref.inputArr[1]*scale_multiplier),(100,0,100),4)
		

	def drawDecisionBoundary(self):
		for point in self.centroidArr:
			for ref in self.centroidArr:

				if(point.inputArr[0] == ref.inputArr[0] and point.inputArr[1] == ref.inputArr[1]):
					continue;#point and ref are the same point

				midPt = dataPoint((point.inputArr[0] + ref.inputArr[0])/2, (point.inputArr[1] + ref.inputArr[1])/2);
				if(point.inputArr[0] != ref.inputArr[0]):
					slope = -1/(float(point.inputArr[1] - ref.inputArr[1])/float(point.inputArr[0] - ref.inputArr[0]));
				else:
					slope = 100000000000; #infinite slope for vertical line
				c = midPt.inputArr[1] - slope*midPt.inputArr[0]; #c -> y = mx+c
				
				x_intercept = int((c*-1)/slope); #when y=0
				y_intercept = int(c);
				if(x_intercept >0 and y_intercept >0):
					cv2.line(self.graph,(0,y_intercept*scale_multiplier),(x_intercept*scale_multiplier,0),(100,0,100),4)
				

	def printCentroids(self):
		print('Centroids computed:');
		for c in self.centroidArr:
			print(c.inputArr);	

