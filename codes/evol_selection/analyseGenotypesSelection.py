import numpy as np
import math
from numba import cuda, int64, jit
from timeit import default_timer as timer
import pickle as pkl
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

@jit
def transformAlignment(genotypes):
    N=int(genotypes.shape[0])
    L=int(genotypes.shape[1])
    alignment=[]
    for pos in range(0,L*64,2):
        posInt = pos//64
        posInInt = pos%64
        variant=[]
        for i in range(N):
            if int(genotypes[i,posInt]) & 2**posInInt == 0 and int(genotypes[i,posInt]) & 2**(posInInt+1) == 0:
                 variant.append("A")
            elif int(genotypes[i,posInt]) & 2**posInInt == 0 and int(genotypes[i,posInt]) & 2**(posInInt+1) > 0:
                 variant.append("C")
            elif int(genotypes[i,posInt]) & 2**posInInt > 0 and int(genotypes[i,posInt]) & 2**(posInInt+1) == 0:
                 variant.append("G")
            else:
                 variant.append("T")
        alignment.append(variant)
    return(alignment)

def writeAlignmentFile(alignment,subindx,filename,S):
    outfile=open(filename,'w')
    outfile.write(str(len(alignment[:][0]))+" "+str(len(alignment))+"\n")
    for i in range(S):
        r="".join([alignment[a][i] for a in range(len(alignment))])
        outfile.write("S"+str(subindx[i])+" "*(100-len("S"+str(subindx[i])))+r+"\n")
    outfile.close()


ID=sys.argv[1]

if os.path.isfile(ID+"/selectionGenotypes4Letters_"+ID+".pkl"):
    infile=open(ID+"/selectionGenotypes4Letters_"+ID+".pkl",'rb')
    subindx=pkl.load(infile)
    N=pkl.load(infile)
    L=pkl.load(infile)
    subgenotypes = pkl.load(infile)
    reference = pkl.load(infile)
    infile.close()
else:
    subgenotypes=np.loadtxt(ID+"/subgenotypes_"+ID+".csv", delimiter=',')
    reference=np.loadtxt(ID+"/reference_"+ID+".csv", delimiter=',')
    subindx=np.loadtxt(ID+"/subindx_"+ID+".csv", delimiter=',')
 
logfile = open(ID+"/"+ID+".log",'r')
logtable = logfile.readlines()
logfile.close()

S=int(subgenotypes.shape[0])

alignment = transformAlignment(subgenotypes)
writeAlignmentFile(alignment,subindx,ID+"/sim_"+ID+".phy",S)

alignment=np.transpose(np.array(alignment))

referenceTransformed = np.transpose(np.array(transformAlignment(np.reshape(reference,(1,reference.shape[0])))))[0,:]

diffArr=[]
for i in range(alignment.shape[0]):
    count=0
    for j in range(alignment.shape[1]):
        if alignment[i,j]!=referenceTransformed[j]:
            count+=1
    diffArr.append(count)
np.savetxt(ID+"/snpsPerStrainRef.txt", np.array(diffArr),fmt='%d', delimiter=',')

diffArr=[]
for i in range(alignment.shape[0]):
    count=0
    for j in range(alignment.shape[1]):
        if alignment[i,j]!=referenceTransformed[j] and list(alignment[:,j]).count(referenceTransformed[j])>0:
            count+=1
    diffArr.append(count)
np.savetxt(ID+"/snpsPerStrainMRCA.txt", np.array(diffArr),fmt='%d', delimiter=',')


pp = PdfPages(ID+'/SFS.pdf')
sfs = np.zeros(S+1, dtype=np.float)
#print(referenceTransformed[:100])
#print(alignment[0,:100])
#print(alignment.shape)

for c in range(alignment.shape[1]):
    column = alignment[:,c]
    freq = S - list(column).count(referenceTransformed[c])
    #print(freq)
    #if freq==50:
        #print(column, referenceTransformed[c])
    if freq:
        sfs[freq] += 1

    #for i in range(1,S+1):
    #    sfs[i] = sfs[i]/(S-(i-1))

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(121)
sc = ax.scatter(range(1,S),sfs[1:-1])
ax.plot(range(1,S), [sfs[1]/i for i in range(1,S)],c="blue",label="1/n")
ax.plot(range(1,S), [sfs[1]/i**2 for i in range(1,S)],c="red",label="1/n^2")
plt.title("SFS")
plt.xlabel("N")
plt.ylabel("frequency spectrum")
plt.legend()
ax = fig.add_subplot(122)
sc = ax.scatter(range(1,S),sfs[1:-1])
ax.plot(range(1,S), [sfs[1]/i for i in range(1,S)],c="blue")
ax.plot(range(1,S), [sfs[1]/i**2 for i in range(1,S)],c="red")
plt.title("SFS")
plt.xlabel("N")
plt.ylabel("frequency spectrum")
plt.gca().set_ylim(1, max(sfs[1:]+100))
plt.gca().set_yscale("log")
plt.savefig(pp, format='pdf')
plt.clf()
pp.close()
