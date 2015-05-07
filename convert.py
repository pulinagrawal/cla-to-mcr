#!/usr/bin/python

import random
import csv
import statistics
import sys
import copy
import math

def generateCLAVector(number,size,sparseness):
    sample_vector=[1]*int(sparseness*size)+[0]*(size-int(sparseness*size))
    vector=[[1]*int(sparseness*size)+[0]*(size-int(sparseness*size))]*number
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

def multiplyMCR(mcr1,mcr2,r):
    mcrR=list()
    for i in range(0,len(mcr1)):
        mcrR.append((mcr1[i]+mcr2[i])%(r[1]-r[0]+1)) #does not account for ranges with negative integers 
    return mcrR

def inverseMCR(mcr,r):
    mcrR=list()
    for i in range(0,len(mcr)):
            mcrR.append((r[1]+1-mcr[i])%r[1]-r[0]+1)
    return mcrR

def sumTable(r):
        circAngle=2*(22/7)
        delTheta=circAngle/(r[1]-r[0]+1)
        table=list()
        for i in range(0,r[1]-r[0]+1):
                table.append(list())
                for j in range(0,r[1]-r[0]+1):
                    y=math.sin(i*delTheta)+math.sin(j*delTheta)
                    x=math.cos(i*delTheta)+math.cos(j*delTheta)
                    if x==0 :
                            table[i].append(1)
                    else:
                            table[i].append(math.atan(y/x))
        return table

def vSum(a,b,r):
        R=0
        modular=r[1]-r[0]+1
        hMod=modular/2
        if a%modular>hMod and b%modular>hMod or a%modular<hMod and b%modular<hMod :
                R=(a+b)/2
        else :
                if a>b :
                    c=(modular-(a-b))/2
                    R=a+c
                else :
                    c=(modular-(b-a))/2
                    R=b+c
        return R

def vectorSum(a,table,r):
        modular=r[1]-r[0]+1
        #circAngle=2*(22/7)
        #delTheta=circAngle/(r[1]-r[0]+1)
        #add=round(table[a][b]/delTheta)
        add=a[0]
        for i in range(1,len(a)):
            add=vSum(add,a[i],r)
        return round(add)%modular

def addMCR(mcrs,r):
    mcrR=list()
    for j in range(0,len(mcrs[0])):
        mcrR.append(0)
        values=[ mcrs[i][j] for i in range(0,len(mcrs)) ]
        table=[1]
        mcrR[j]= vectorSum(values,table,r)
    return mcrR


def convertCLAtoMCR(vector,conversionSet,r,indexSet):
    mcrVector=list()
    sp_vector=sparsify(vector)
    selConvSet=[ conversionSet[i] for i in sp_vector ]
    return addMCR(selConvSet,r)
    '''... This was previous implementation
        mcrVector=conversionSet[sp_vector[0]]
        for i in range(1,len(sp_vector)):
            mcrVector=addMCR(mcrVector,conversionSet[sp_vector[i]],r)
        return mcrVector
    ...'''

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

def main(nV,sparseness,noise):
    number_of_vectors=nV
    dimension_of_cla_vectors=1024
    dimension_of_mcr_vectors=1024
    r=[0,15]
    mcrV=list()

    claV=generateCLAVector(number_of_vectors,dimension_of_cla_vectors,sparseness)
    print( distance(claV[0],claV[1],[0,1]))

    convSet=generateConversionSet(dimension_of_cla_vectors,dimension_of_mcr_vectors,r)
    idxSet=generateConversionSet(dimension_of_cla_vectors,dimension_of_cla_vectors,r)
    print (distance(convertCLAtoMCR(claV[0],convSet,r,idxSet),convertCLAtoMCR(claV[1],convSet,r,idxSet),r))

    for i in range(0,number_of_vectors):
        mcrV.append(convertCLAtoMCR(claV[i],convSet,r,idxSet))
    
    CLA_dist=list()
    MCR_dist=list()
    random_MCR_dist=list()
    CLA_noisy_dist=list()
    MCR_noisy_dist=list()

    avg_CLA_dist=0
    avg_MCR_dist=0
    avg_random_MCR_dist=0
    avg_CLA_noisy_dist=0
    avg_MCR_noisy_dist=0
    
    for i in range(0,number_of_vectors):
        for j in range(i+1,number_of_vectors):
            CLA_dist.append(distance(claV[i],claV[j],[0,1]))
            MCR_dist.append(distance(mcrV[i],mcrV[j],r))
            random_MCR_dist.append(distance(convSet[i],convSet[j],r))

    for i in range(0,number_of_vectors):
        noisyV=addNoise(claV[i],noise)
        CLA_noisy_dist.append(distance(claV[i],noisyV,[0,1]))
        MCR_noisy_dist.append(distance(mcrV[i],convertCLAtoMCR(noisyV,convSet,r,idxSet),r))


    f = open('dist_datapoints','w')
    csvwriter=csv.writer(f)
    csvwriter.writerow(CLA_dist)
    csvwriter.writerow(MCR_dist)
    csvwriter.writerow(CLA_noisy_dist)
    csvwriter.writerow(MCR_noisy_dist)
    f.close()

    avg_CLA_dist=statistics.mean(CLA_dist)
    avg_MCR_dist=statistics.mean(MCR_dist)
    avg_MCR_noisy_dist=statistics.mean(MCR_noisy_dist)
    avg_CLA_noisy_dist=statistics.mean(CLA_noisy_dist)
    avg_random_MCR_dist=statistics.mean(random_MCR_dist)

    print ("Vectors used="+str(nV))
    print ("Average random MCR distance="+str(avg_random_MCR_dist) ) #average distance between any random MCR vectors
    print ("Average CLA Distance="+str(avg_CLA_dist) )#average distance of combination of all points(CLA vectors) in CLA space
    print ("Average MCR Distance="+str(avg_MCR_dist) )#average distance of combination of all points(MCR vectors) in MCR space
    print ("Average CLA Distance in "+str(noise)+" noisy CLA="+str(avg_CLA_noisy_dist) )#average distance between MCR projection of CLA vector and its noisy version
    print ("Average MCR Distance from "+str(noise)+" noisy CLA="+str(avg_MCR_noisy_dist) )#average distance between MCR projection of CLA vector and its noisy version

    sdv_CLA_dist=statistics.stdev(CLA_dist)
    sdv_MCR_dist=statistics.stdev(MCR_dist)
    sdv_MCR_noisy_dist=statistics.stdev(MCR_noisy_dist)
    sdv_CLA_noisy_dist=statistics.stdev(CLA_noisy_dist)
    sdv_random_MCR_dist=statistics.stdev(random_MCR_dist)

    print ("Standard Deviation random MCR Distance="+str(sdv_random_MCR_dist) )#std dev distance of random MCR vectors
    print ("Standard Deviation CLA Distance="+str(sdv_CLA_dist) )#std dev distance of combijation of all points(CLA vectors) in CLA space
    print ("Standard Deviation MCR Distance="+str(sdv_MCR_dist) )#std dev distance of combination of all points(MCR vectors) in MCR space
    print ("Standard Deviation CLA Distance in "+str(noise)+" noisy CLA="+str(sdv_CLA_noisy_dist) )#std dev distance between MCR projection of CLA vector and its noisy version
    print ("Standard Deviation MCR Distance from "+str(noise)+" noisy CLA="+str(sdv_MCR_noisy_dist)) #std dev distance between MCR projection of CLA vector and its noisy version

    return
