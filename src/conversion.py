
region_size = 100
region_sparsity = .02

test_size = 12

import convert as conv
import numpy as np
import isdm

def generate_random_mcrs(number, size):
    return

def to_mcr(sdrs, convSet):
    mcrs = []
    for sdr in sdrs:
        v = conv.sparsify(sdr)
        sel_conv_set = [convSet[i] for i in v]
        mcrs.append(sum(sel_conv_set[1:], sel_conv_set[0]))

    return mcrs

sdrs = conv.generateCLAVector(test_size, region_size, region_sparsity)
print(sdrs)

convSet = [isdm.MCRVector.random_vector() for i in range(region_size)]
mcrs = to_mcr(sdrs, convSet)

def avg_mcrs_dist(mcrs):
    dist = []
    for i in range(0, len(mcrs)):
        for j in range(i+1, len(mcrs)):
            dist.append(mcrs[i].distance(mcrs[j]))

    return np.mean(dist)

print('Tests if different vectors when converted are far apart')
print(avg_mcrs_dist(mcrs))
print(avg_mcrs_dist(convSet))
print(isdm.IntegerSDM.access_sphere_radius())


print('Tests if similar vectors when converted are close')

noisy_sdrs = [conv.addNoise(sdr, .2) for sdr in sdrs]
noisy_mcrs = to_mcr(noisy_sdrs, convSet)

dist = []
for i in range(len(mcrs)):
    dist.append(mcrs[i].distance(noisy_mcrs[i]))

print(np.mean(dist))
'''
print('Test if we can get back the sdr from mcr')
#store in isdm all vectors
pam = isdm.IntegerSDM(10000)
print('ISDM created')
for mcr in mcrs:
    pam.write(mcr)
print('mcrs written')
for vector in convSet:
    pam.write(vector)

print('convSet written')
#probe a vector for all the vectors in convSet
#create sdr
recon_sdrs = []
for mcr in mcrs:
    sdr = [0]*region_size
    for i, vector in enumerate(convSet):
        if pam.read(mcr*(not vector)) is not None:
            print('read a 1')
            sdr[i] = 1
    recon_sdrs.append(sdr)
    print('vector reconstructed')

#compare original and recon
error = []
for sdr, rsdr in zip(sdrs, recon_sdrs):
    error.append(sum(np.not_equal(sdr, rsdr)))

print('Average Error: ', np.mean(error))
print('Test if we can get back the sdr from MCR when ISDM is filled')

print('Test if we can get back the SDR from complex MCR')

print('Test if we can get back the SDR from complex MCR when ISDM is filled')
'''
