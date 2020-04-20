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
def evolve(rng_states,gen,numberGenerations,genotypes,reference,binomcdf,fitnessarr,F,s):
    cellThreadID = cuda.grid(1)
    genoblocks = genotypes.shape[1]/2
    L=64*genoblocks
    N=genotypes.shape[0]
    numberLocalCells = 1
    for numbIt in range(numberGenerations):
        for localI in range(numberLocalCells):
            cellID = (cellThreadID*numberLocalCells) + localI
            if cellID < N:
               if (gen+numbIt) % 2 == 1:
                   targetoffset = 0
                   sourceoffset = int(genotypes.shape[1]/2)
               else:
                   targetoffset = int(genotypes.shape[1]/2)
                   sourceoffset = 0
               #choose a random parent (according to fitnesses)
               r=F[0]*rnd.xoroshiro128p_uniform_float32(rng_states, cellThreadID)
               for i in range(N):
                   if r<fitnessarr[i]:
                       parent=i
                       break
                   else:
                       r-=fitnessarr[i]
               for i in range(genoblocks):
                   genotypes[cellID,i+targetoffset]=genotypes[parent,i+sourceoffset]
               #add mutations
               r=rnd.xoroshiro128p_uniform_float32(rng_states, cellThreadID)
               for k in range(100):
                   if r < binomcdf[k]:
                       break
               numberMutations=k
               for i in range(numberMutations):
                   pos = 2*int(math.floor(L/2*rnd.xoroshiro128p_uniform_float32(rng_states, cellThreadID)))
                   posInt = pos//64
                   posInInt = pos%64
                   r=rnd.xoroshiro128p_uniform_float32(rng_states, cellThreadID)
                   if r < 1/4:
                       #transverion (first bit with probability 1/2)
                       if r < 1/8:
                           genotypes[cellID,posInt+targetoffset]=genotypes[cellID,posInt+targetoffset] ^ 2**posInInt
                       #second bit has to be flipped
                       genotypes[cellID,posInt+targetoffset]=genotypes[cellID,posInt+targetoffset] ^ 2**(posInInt+1)
                   else:
                       #transition (00<->10 or 01<->11)
                       genotypes[cellID,posInt+targetoffset]=genotypes[cellID,posInt+targetoffset] ^ 2**posInInt
               snpCount=0
               for posInt in range(genoblocks):
                   intAnd=(genotypes[cellID,posInt+targetoffset] ^ reference[posInt])
                   for posInInt in range(0,64,2):
                       if intAnd & 2**posInInt > 0 or intAnd & 2**(posInInt+1) > 0:
                           snpCount+=1
               fitnessarr[cellID]=s**snpCount
        cuda.syncthreads()

from scipy.optimize import fsolve
def mufunct(mu,N,S):
    return np.prod([1/(1+2*mu*N/(k-1)) for k in range(2,S+1)])-0.9

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="the seed of the simulation (default=1)", default=0)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action='count')
parser.add_argument("--threadsperblock", type=int, help="the number of threads per block on the GPU; N=threadsperblock*blocks ", default=60)
parser.add_argument("--blocks", type=int, help="the number of blocks; N=threadsperblock*blocks", default=60)
parser.add_argument("-S", "--samplesize", type=int, help="the size of sample", default=50)
parser.add_argument("-p", "--fractionvariable", type=float, help="the fraction of positions which are expected to be variable", default=0.9)
parser.add_argument("-s", "--selectionc", type=float, help="multiplicative cost of a single mutation", default=1.0)
parser.add_argument("-I", "--ID", help="the ID of the simulation run", required=True)
parser.add_argument("-L", "--genomelength", type=int, help="length of genome in 64 bp blocks", default=40000)
args = parser.parse_args()

if args.verbose:
    print("number of cores: "+str(numba.config.NUMBA_DEFAULT_NUM_THREADS))

ID=args.ID
if not os.path.exists(ID):
    os.makedirs(ID)
L=args.genomelength
n=64*L
threadsperblock = args.threadsperblock
blocks = args.blocks
N=threadsperblock*blocks
S=args.samplesize
s=args.selectionc
p2=fsolve(mufunct,0.000001,args=(N,S))[0]
if args.verbose:
    print("p2: "+str(p2))
binomcdf = np.zeros((100),dtype=np.float32)
for k in range(100):
    binomcdf[k]=binom.cdf(k, n, p2)
SEED=args.seed
logfile=open(ID+"/"+ID+".log",'w')
logfile.write("L: "+str(L)+"\n")
logfile.write("n: "+str(n)+"\n")
logfile.write("N: "+str(N)+"\n")
logfile.write("S: "+str(S)+"\n")
logfile.write("s: "+str(s)+"\n")
logfile.write("p2: "+str(p2)+"\n")
logfile.write("seed: "+str(SEED)+"\n")
NUMCYCLES=8*N
subindx=range(0,S)
genotypes = np.array(np.reshape(np.array(list(np.random.uniform(0,2**64, L*2))*2*N),(N,L*4)), dtype=np.uint64)
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
evolve[blocks, threadsperblock](rng_states,0,1,devgenotypes,devreference,devbinomcdf,devfitnessarr,F,s)
sumF[1,1](devfitnessarr,F)
evolve_time=timer()-start
if args.verbose:
    print("first iteration: "+str(evolve_time))
logfile.write("first iteration: "+str(evolve_time)+"\n")

start = timer()
for i in range(1,1001):
    #start = timer()
    evolve[blocks, threadsperblock](rng_states,i,1,devgenotypes,devreference,devbinomcdf,devfitnessarr,F,s)
    #fitnessarr=devfitnessarr.copy_to_host()
    #print(F)
    sumF[1,1](devfitnessarr,F)
    #print(F)
    #print(fitnessarr)
    #F=np.sum(fitnessarr)
#print(F)
cuda.synchronize()
evolve_time=timer()-start
if args.verbose:
    print("1000 more iterations: "+str(evolve_time))
logfile.write("1000 more iterations: "+str(evolve_time)+"\n")

start = timer()
for i in range(1001,NUMCYCLES):
    evolve[blocks, threadsperblock](rng_states,i,1,devgenotypes,devreference,devbinomcdf,devfitnessarr,F,s)
    #fitnessarr=devfitnessarr.copy_to_host()
    #F=np.sum(fitnessarr)
    #cuda.synchronize()
    sumF[1,1](devfitnessarr,F)
Fl=F.copy_to_host()
print(Fl)
#fitnessarr=devfitnessarr.copy_to_host()
#print(fitnessarr)
cuda.synchronize()
evolve_time=timer()-start
if args.verbose:
    print("remaining "+str(NUMCYCLES-1000)+" cylcles: "+str(evolve_time)+"\n")
logfile.write("remaining "+str(NUMCYCLES-1000)+" iterations: "+str(evolve_time)+"\n")

cuda.synchronize()
start = timer()
genotypes = devgenotypes.copy_to_host()
evolve_time=timer()-start
if args.verbose:
    print("copying of data to memory: "+str(evolve_time)+"\n")
logfile.write("copying of data to memory: "+str(evolve_time)+"\n")

outfile=open(ID+"/selectionGenotypes4Letters_"+ID+".pkl",'wb')
pkl.dump(subindx, outfile)
pkl.dump(N, outfile)
pkl.dump(L, outfile)
subset = [(i in subindx) for i in range(N) ]
subgenotypes = genotypes[subset,:(2*L)]
pkl.dump(subgenotypes, outfile)
pkl.dump(reference, outfile)
outfile.close()
logfile.close()
