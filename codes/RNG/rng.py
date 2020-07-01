import sys 
import numpy as np
import math
import numba
from numba import cuda, int64, jit
from numba.cuda import random as rnd
from timeit import default_timer as timer
from scipy.stats import binom, poisson
import pickle as pkl
#np.set_printoptions(threshold=np.nan)
import argparse
import os

@cuda.jit
def sumF(fitnessarr,F):
    F[0]=0
    for i in range(fitnessarr.shape[0]):
        F[0]+=fitnessarr[i]


@cuda.jit
def evolve(rng_states,gen,genotypes,reference,binomcdf,fitnessarr,F):
    genoblocks = genotypes.shape[1]/2
    cellID = cuda.grid(1)
    #this is how you would get a multinomially distributed random variable
    r=F[0]*rnd.xoroshiro128p_uniform_float32(rng_states, cellID)
    for i in range(len(fitnessarr)):
        if r<fitnessarr[i]:
            parent=i
            break
        else:
            r-=fitnessarr[i]
    #this is for a binomially distibuted random variable, if here doing a serial search should be fine. However, in some situations binary a binary search should be performed 
    r=rnd.xoroshiro128p_uniform_float32(rng_states, cellID)
    for numberMutations in range(100):
        if r < binomcdf[numberMutations]:
            break
    for i in range(numberMutations):
        #this is for a uniformly distributed integer
        pos = 2*int(math.floor(L/2*rnd.xoroshiro128p_uniform_float32(rng_states, cellID)))
        posInt = pos//64
        posInInt = pos%64
        #this is just a simple uniformly distibuted float
        r=rnd.xoroshiro128p_uniform_float32(rng_states, cellID)
        if r < 1/4:
             #transverion (first bit with probability 1/2)
            if r < 1/8:
                genotypes[cellID,posInt]=genotypes[cellID,posInt] ^ 2**posInInt
                #second bit has to be flipped
                genotypes[cellID,posInt]=genotypes[cellID,posInt] ^ 2**(posInInt+1)
        else:
            #transition (00<->10 or 01<->11)
            genotypes[cellID,posInt]=genotypes[cellID,posInt] ^ 2**posInInt
    snpCount=0
    for posInt in range(genoblocks):
        intAnd=(genotypes[cellID,posInt] ^ reference[posInt])
        for posInInt in range(0,64,2):
            if intAnd & 2**posInInt > 0 or intAnd & 2**(posInInt+1) > 0:
                snpCount+=1
    #here we prepare the multinomial for the next interation
    fitnessarr[cellID]=snpCount

L=1000
n=64*L
threadsperblock = 30
blocks = 30
N=threadsperblock*blocks
fractionvariable=0.9
p2=0.00001
binomcdf = np.zeros((100),dtype=np.float32)
for k in range(100):
    binomcdf[k]=binom.cdf(k, n, p2)
SEED=1
genotypes = np.array(np.reshape(np.array(list(np.random.uniform(0,2**64, L))*2*N),(N,L*2)), dtype=np.uint64)
devgenotypes = cuda.to_device(genotypes)
reference = np.copy(genotypes[0,:])
devreference = cuda.to_device(reference)
devbinomcdf = cuda.to_device(binomcdf)
fitnessarr = np.ones(N)
F=np.sum(fitnessarr)
devfitnessarr=cuda.to_device(fitnessarr)
rng_states = rnd.create_xoroshiro128p_states(threadsperblock * blocks, seed=SEED)
F=cuda.device_array((1))
sumF[1,1](devfitnessarr,F)

start = timer()
evolve[blocks, threadsperblock](rng_states,0,devgenotypes,devreference,devbinomcdf,devfitnessarr,F)
sumF[1,1](devfitnessarr,F)
cuda.synchronize()
evolve_time=timer()-start
print("first iteration: "+str(evolve_time))
fitnessarr=devfitnessarr.copy_to_host()
print(fitnessarr)
print(F)

start = timer()
evolve[blocks, threadsperblock](rng_states,1,devgenotypes,devreference,devbinomcdf,devfitnessarr,F)
sumF[1,1](devfitnessarr,F)
cuda.synchronize()
evolve_time=timer()-start
print("second iteration: "+str(evolve_time))
fitnessarr=devfitnessarr.copy_to_host()
print(fitnessarr)
print(F)

