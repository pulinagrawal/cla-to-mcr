#!/usr/bin/python

import random
import copy
import math

def generateCLAVector(number,size):
	sample_vector=[1]*int(.02*size)+[0]*(size-int(.02*size))
	vector=[[1]*int(.02*size)+[0]*(size-int(.02*size))]*number
	for j in range(0,number):
		vector[j]=sample_vector[:]
		random.shuffle(vector[j])
	return vector

def sparsify(vector):
	sparse_vector=list()
	for i in range(0,len(vector)):
		if vector[i]==1:
			sparse_vector.append(i)
	return sparse_vector

def generateConversionSet(number,size,r):
	conversionSet=list()
	for j in range(0,number):
		conversionSet.append(list())
		for i in range(0,size):
			conversionSet[j].append(random.randint(r[0],r[1]))
	return conversionSet

def addMCR(mcr1,mcr2,r):
	mcrR=list()
	for i in range(0,len(mcr1)):
		mcrR.append((mcr1[i]+mcr2[i])%(r[1]-r[0]+1)) #does not account for ranges with negative integers 
	return mcrR

def convertCLAtoMCR(vector,conversionSet,r):
	mcrVector=list()
	sp_vector=sparsify(vector)
	mcrVector=conversionSet[sp_vector[0]]
	for i in range(1,len(sp_vector)):
		mcrVector=addMCR(mcrVector,conversionSet[i],r)
	return mcrVector

def distance(vector1,vector2):
	distance=0
	for i in range(0,len(vector1)):
		distance+=math.fabs(vector1[i]-vector2[i])
	return distance


claV=generateCLAVector(3,100)
print claV
print distance(claV[0],claV[1])
r=[0,15]
convSet=generateConversionSet(100,100,r)
print distance(convertCLAtoMCR(claV[0],convSet,r),convertCLAtoMCR(claV[1],convSet,r))
