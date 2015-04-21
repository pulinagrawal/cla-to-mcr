#!/usr/bin/python

import random
import sys
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
    #number - number of mcr vectors = dimensions in cla vector
    #size - length of each mcr vector = dimensionality of mcr space
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

def distance(vector1,vector2,r):
	distance=0
	modular=r[1]-r[0]+1
	for i in range(0,len(vector1)):
		distance+=min((vector1[i]-vector2[i])%modular,(vector2[i]-vector1[i])%modular)
	return distance

def addNoise(vector,pct):
   ones_index=sparsify(vector)    
   selected=random.sample(ones_index,int(pct*len(ones_index)))  
   zeroes_index=[ range(0,len(vector))[i] for i in range(0,len(vector)) if i not in ones_index] 
   selected_zeroes=random.sample(zeroes_index,int(pct*len(ones_index)))
   selected.extend(selected_zeroes)
   noisy=vector[:]
   for i in selected:
        noisy[i]^=1 # there are many ways to flip 1->0 and 0->1 like var=not var, var = (0,1)[var], var=1-var
   return noisy

def main(nV):
    number_of_vectors=nV
    dimension_of_cla_vectors=1024
    dimension_of_mcr_vectors=1024
    r=[0,15]
    noise=.1

    mcrV=list()
    claV=generateCLAVector(number_of_vectors,dimension_of_cla_vectors)
    print distance(claV[0],claV[1],[0,1])
    convSet=generateConversionSet(dimension_of_cla_vectors,dimension_of_mcr_vectors,r)
    
    print distance(convertCLAtoMCR(claV[0],convSet,r),convertCLAtoMCR(claV[1],convSet,r),r)
    for i in range(0,number_of_vectors):
        mcrV.append(convertCLAtoMCR(claV[i],convSet,r))
    
    avg_CLA_dist=0
    avg_MCR_dist=0
    avg_CLA_noisy_dist=0
    avg_MCR_noisy_dist=0
    
    for i in range(0,number_of_vectors):
        for j in range(i+1,number_of_vectors):
            avg_CLA_dist+=distance(claV[i],claV[j],[0,1])
            avg_MCR_dist+=distance(mcrV[i],mcrV[j],r)
    
    combinations=(number_of_vectors*(number_of_vectors-1))/2
    avg_CLA_dist/=combinations
    avg_MCR_dist/=combinations

    for i in range(0,number_of_vectors):
    	    noisyV=addNoise(claV[i],noise)
            avg_CLA_noisy_dist+=distance(claV[i],noisyV,[0,1])
            avg_MCR_noisy_dist+=distance(mcrV[i],convertCLAtoMCR(noisyV,convSet,r),r)

    avg_MCR_noisy_dist/=number_of_vectors 
    avg_CLA_noisy_dist/=number_of_vectors 

    print "Average CLA Distance="+str(avg_CLA_dist) #average distance of combination of all points(CLA vectors) in CLA space

    print "Average MCR Distance="+str(avg_MCR_dist) #average distance of combination of all points(MCR vectors) in MCR space
    print "Average CLA Distance in "+str(noise)+" noisy CLA="+str(avg_CLA_noisy_dist) #average distance between MCR projection of CLA vector and its noisy version
    print "Average MCR Distance from "+str(noise)+" noisy CLA="+str(avg_MCR_noisy_dist) #average distance between MCR projection of CLA vector and its noisy version
    return
