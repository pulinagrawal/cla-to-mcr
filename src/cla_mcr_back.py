#!/usr/bin/env python
# coding: utf-8
import tqdm



# In[22]:


import random

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
        noisy[i] ^= 1  # there are many ways to flip 1->0 and 0->1 like var=not var, var = (0,1)[var], var=1-var
    return noisy


def main(number_of_vectors=30, threshold_std_multiplier=3.83,
         dimension_of_mcr_vectors=1024, dimension_of_cla_vectors=1024,
         sparseness=0.02, r=[0, 15]):
    # In[27]:

    from src import isdm
    nV = number_of_vectors

    isdm.configure(dimension_of_mcr_vectors, contains_std_threshold=threshold_std_multiplier)
    mcrV = list()

    # In[35]:

    nV = number_of_vectors

    # In[28]:

    claV = generateCLAVector(number_of_vectors, dimension_of_cla_vectors, sparseness)

    # In[29]:

    convSet = generateConversionSet(dimension_of_cla_vectors, dimension_of_mcr_vectors, r)


    # In[30]:

    def conv_sdr_to_mcr(sdr, convSet):
        v = sparsify(sdr)
        sel_conv_set = [convSet[i] for i in v]
        return isdm.MCRVector._addMCR(sel_conv_set)

    mcr_conv_set = [isdm.MCRVector(convSetv) for convSetv in convSet]

    mcrV = list()
    for i in tqdm.tqdm(range(0, number_of_vectors)):
        mcrV.append(conv_sdr_to_mcr(claV[i], mcr_conv_set))

    # In[33]:

    import numpy as np

    # In[36]:


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

    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    print("Average Precision" + str(avg_precision) + '±' + str(np.std(precision)))
    print("Average Recall" + str(avg_recall) + '±' + str(np.std(recall)))

    return 2*precision*recall/(precision+recall)


if __name__ == '__main__':
    main()
