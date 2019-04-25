#!/usr/bin/env python
# coding: utf-8
import tqdm

# In[ ]:

r = [0, 15]
theta = 1
circAngle = 44.0 / 7
delTheta = circAngle / (r[1] - r[0] + 1)
print(circAngle)
print(delTheta)
print(round(theta / delTheta))

# In[ ]:


# In[22]:


import random
import csv
import sys
import copy
import math


class Vector:
    theta = 0
    mag = 0
    circAngle = 44.0 / 7

    def __init__(self, val, r):
        delTheta = circAngle / (r[1] - r[0] + 1)
        self.mag = 1
        self.theta = val * delTheta

    def modValue(self, r):
        delTheta = circAngle / (r[1] - r[0] + 1)
        return round(self.theta / delTheta)

    def setVector(self, val, r):
        delTheta = circAngle / (r[1] - r[0] + 1)
        self.mag = 1
        self.theta = val * delTheta

    def setVector(self, val, mag, r):
        self.mag = mag
        self.theta = val * delTheta

    def addVector(self, v):
        R = Vector(0, [0, 15])
        y = v.mag * math.sin(v.theta) + self.mag * math.sin(self.theta)
        x = v.mag * math.cos(v.theta) + self.mag * math.cos(self.theta)
        if x == 0:
            if y > 0:
                R.theta = (circAngle / 4)
            else:
                R.theta = (circAngle * 3 / 4)
        else:
            R.theta = math.atan(abs(y / x))

        if y < 0 and x > 0:
            R.theta = circAngle - R.theta
        elif y < 0 and x < 0:
            R.theta = circAngle / 2 + R.theta
        elif y > 0 and x < 0:
            R.theta = circAngle / 2 - R.theta

        if x == 0 and y == 0:
            # insert the chance logic
            R.theta = self.theta + circAngle / 4

        R.mag = (y ** 2 + x ** 2) ** 0.5
        return R


# In[23]:


def generateCLAVector(number, size, sparseness):
    sample_vector = [1] * int(sparseness * size) + [0] * (size - int(sparseness * size))
    vector = [[1] * int(sparseness * size) + [0] * (size - int(sparseness * size))] * number
    for j in tqdm.tqdm(range(0, number)):
        vector[j] = sample_vector[:]
        random.shuffle(vector[j])
    return vector


def sparsify(vector):
    sparse_vector = list()
    for i in range(0, len(vector)):
        if vector[i] == 1:
            sparse_vector.append(i)
    return sparse_vector


def generateConversionSet(number, size, r):
    # number - number of mcr vectors = dimensions in cla vector
    # size - length of each mcr vector = dimensionality of mcr space
    conversionSet = list()
    for j in tqdm.tqdm(range(0, number)):
        conversionSet.append(list())
        for i in range(0, size):
            conversionSet[j].append(random.randint(r[0], r[1]))
    return conversionSet


def multiplyMCR(mcr1, mcr2, r):
    mcrR = list()
    for i in range(0, len(mcr1)):
        mcrR.append((mcr1[i] + mcr2[i]) % (
                    r[1] - r[0] + 1))  # does not account for ranges with negative integers
    return mcrR


def inverseMCR(mcr, r):
    mcrR = list()
    for i in range(0, len(mcr)):
        mcrR.append((r[1] + 1 - mcr[i]) % r[1] - r[0] + 1)
    return mcrR


def vectorSum(a, r, isTrig):
    global gf
    modular = r[1] - r[0] + 1
    circAngle = 2 * (22 / 7)
    delTheta = circAngle / (r[1] - r[0] + 1)
    add = Vector(a[0], r)
    for i in range(1, len(a)):
        v = Vector(a[i], r)
        add = add.addVector(v)
    return (add.modValue(r)) % modular


def addMCR(mcrs, r):
    mcrR = list()
    for j in range(0, len(mcrs[0])):
        mcrR.append(0)
        values = [mcrs[i][j] for i in range(0, len(mcrs))]
        mcrR[j] = vectorSum(values, r, True)
    return mcrR


def convertCLAtoMCR(vector, conversionSet, r, indexSet):
    mcrVector = list()
    sp_vector = sparsify(vector)
    selConvSet = [conversionSet[i] for i in sp_vector]
    return addMCR(selConvSet, r)
    '''... This was previous implementation
        mcrVector=conversionSet[sp_vector[0]]
        for i in range(1,len(sp_vector)):
            mcrVector=addMCR(mcrVector,conversionSet[sp_vector[i]],r)
        return mcrVector
    ...'''


def distance(vector1, vector2, r):
    distance = 0
    modular = r[1] - r[0] + 1
    for i in range(0, len(vector1)):
        distance += min((vector1[i] - vector2[i]) % modular, (vector2[i] - vector1[i]) % modular)
    return distance


def addNoise(vector, pct):
    ones_index = sparsify(vector)
    selected = random.sample(ones_index, int(pct * len(ones_index)))
    zeroes_index = [range(0, len(vector))[i] for i in range(0, len(vector)) if i not in ones_index]
    selected_zeroes = random.sample(zeroes_index, int(pct * len(ones_index)))
    selected.extend(selected_zeroes)
    noisy = vector[:]
    for i in selected:
        noisy[
            i] ^= 1  # there are many ways to flip 1->0 and 0->1 like var=not var, var = (0,1)[var], var=1-var
    return noisy


def analyzeAddition(vectors, r, times):
    # TODO add definition
    return


def main(number_of_vectors=30, threshold_std_multiplier=3.83,
         dimension_of_mcr_vectors=1024, dimension_of_cla_vectors=1024,
         sparseness=0.02, noise=0.2, r=[0, 15]):
    # In[27]:

    from src import isdm
    nV = number_of_vectors

    isdm.configure(dimension_of_mcr_vectors, contains_std_threshold=threshold_std_multiplier)
    mcrV = list()

    # In[35]:

    nV = number_of_vectors

    # In[28]:

    claV = generateCLAVector(number_of_vectors, dimension_of_cla_vectors, sparseness)
    print(distance(claV[0], claV[1], [0, 1]))

    # In[29]:

    convSet = generateConversionSet(dimension_of_cla_vectors, dimension_of_mcr_vectors, r)

    # print (distance(convertCLAtoMCR(claV[0],convSet,r,idxSet),convertCLAtoMCR(claV[1],convSet,r,idxSet),r))

    # In[30]:

    def conv_sdr_to_mcr(sdr, convSet):
        v = sparsify(sdr)
        sel_conv_set = [convSet[i] for i in v]
        return isdm.MCRVector._addMCR(sel_conv_set)

    mcr_conv_set = [isdm.MCRVector(convSetv) for convSetv in convSet]

    mcrV = list()
    for i in tqdm.tqdm(range(0, number_of_vectors)):
        # print('Method 1')
        # mcrV.append(convertCLAtoMCR(claV[i],convSet,r,idxSet))
        # print('Method 2')
        mcrV.append(conv_sdr_to_mcr(claV[i], mcr_conv_set))

    # In[33]:

    import numpy as np

    # In[36]:

    CLA_dist = list()
    MCR_dist = list()
    random_MCR_dist = list()
    CLA_noisy_dist = list()
    MCR_noisy_dist = list()

    avg_CLA_dist = 0
    avg_MCR_dist = 0
    avg_random_MCR_dist = 0
    avg_CLA_noisy_dist = 0
    avg_MCR_noisy_dist = 0

    for i in range(0, number_of_vectors):
        for j in range(i + 1, number_of_vectors):
            CLA_dist.append(distance(claV[i], claV[j], [0, 1]))
            MCR_dist.append(mcrV[i] % mcrV[j])
            random_MCR_dist.append(mcr_conv_set[i] % mcr_conv_set[j])

    for i in tqdm.tqdm(range(0, number_of_vectors)):
        noisyV = addNoise(claV[i], noise)
        CLA_noisy_dist.append(distance(claV[i], noisyV, [0, 1]))
        MCR_noisy_dist.append(mcrV[i] % conv_sdr_to_mcr(noisyV, mcr_conv_set))

    f = open('dist_datapoints', 'w')
    csvwriter = csv.writer(f)
    csvwriter.writerow(CLA_dist)
    csvwriter.writerow(MCR_dist)
    csvwriter.writerow(CLA_noisy_dist)
    csvwriter.writerow(MCR_noisy_dist)
    f.close()

    avg_CLA_dist = np.mean(CLA_dist)
    avg_MCR_dist = np.mean(MCR_dist)
    avg_MCR_noisy_dist = np.mean(MCR_noisy_dist)
    avg_CLA_noisy_dist = np.mean(CLA_noisy_dist)
    avg_random_MCR_dist = np.mean(random_MCR_dist)


    sdv_CLA_dist = np.std(CLA_dist)
    sdv_MCR_dist = np.std(MCR_dist)
    sdv_MCR_noisy_dist = np.std(MCR_noisy_dist)
    sdv_CLA_noisy_dist = np.std(CLA_noisy_dist)
    sdv_random_MCR_dist = np.std(random_MCR_dist)

    print ("Vectors used=" + str(nV))
    print ("Average random MCR distance=" + str(avg_random_MCR_dist) + '±' + str(sdv_random_MCR_dist))  # average distance between any random MCR vectors
    print ("Average CLA Distance=" + str(avg_CLA_dist) + '±' + str(sdv_CLA_dist))  # average distance of combination of all points(CLA vectors) in CLA space
    print ("Average MCR Distance=" + str( avg_MCR_dist) + '±' + str(sdv_MCR_dist))  # average distance of combination of all points(MCR vectors) in MCR space
    print ("Average CLA Distance in " + str(noise) + " noisy CLA=" + str(avg_CLA_noisy_dist) + '±' + str(sdv_CLA_noisy_dist))
    # average distance between MCR projection of CLA vector and its noisy version
    print ("Average MCR Distance from " + str(noise) + " noisy MCR=" + str(avg_MCR_noisy_dist) + '±' + str(sdv_MCR_noisy_dist))
    # average distance between MCR projection of CLA vector and its noisy version

    # In[72]:

    # In[68]:


    # In[73]:

    precision = np.array([0.] * number_of_vectors)
    recall = np.array([0.] * number_of_vectors)
    for i in tqdm.tqdm(range(0, number_of_vectors)):
        v = np.array([], dtype=int)
        for j in range(0, dimension_of_cla_vectors):
            if mcr_conv_set[j] in mcrV[i]:
                v = np.append(v, j)

        tp = len(np.intersect1d(np.array(sparsify(claV[i])), v))
        fp = len(np.setdiff1d(v, np.array(sparsify(claV[i]))))
        fn = len(np.setdiff1d(np.array(sparsify(claV[i])), v))

        precision[i] = float(tp) / (tp + fp)
        recall[i] = float(tp) / (tp + fn)

    print("Average Precision" + str(np.mean(precision)) + '±' + str(np.std(precision)))
    print("Average Recall" + str(np.mean(recall)) + '±' + str(np.std(recall)))



if __name__ == '__main__':
    main()
