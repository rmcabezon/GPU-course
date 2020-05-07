import os
import math
import numpy as np
from matplotlib import pyplot as plt

cuda = np.loadtxt('timings/time.cuda.txt')
target = np.loadtxt('timings/time.target.txt')
threads = np.loadtxt('timings/time.threads.txt')
order = np.loadtxt('timings/time.order.txt')
naive = np.loadtxt('timings/time.naive.txt')

l = len(cuda)
n = [64 * 2**i for i in range(0, l)]

print(l, n)

plt.figure(figsize=(5,4))
if cuda.any(): plt.plot(n[0:len(cuda)], cuda[:,1], marker='x', linestyle='-', linewidth=2, markersize=6, color='C0', label=r"CUDA")#($f_{child} = 2, n_{p} = 64$)")
if target.any(): plt.plot(n[0:len(target)], target[:,1], marker='v', linestyle='-', linewidth=2, markersize=6, color='C1', label=r"OpenMP Offloading")#($f_{child} = 2, n_{p} = 64$)")
if threads.any(): plt.plot(n[0:len(threads)], threads[:,1], marker='o', linestyle='-', linewidth=2, markersize=6, color='C2', label=r"OpenMP Threading")#($f_{child} = 2, n_{p} = 64$)")
if order.any(): plt.plot(n[0:len(order)], order[:,1], marker='>', linestyle='-', linewidth=2, markersize=6, color='C4', label=r"Row-Major")#($f_{child} = 2, n_{p} = 64$)")
if naive.any(): plt.plot(n[0:len(naive)], naive[:,1], marker='<', linestyle='-', linewidth=2, markersize=6, color='C5', label=r"Column-Major")#($f_{child} = 2, n_{p} = 64$)")
plt.legend(loc='upper left')
plt.xscale('log')
plt.yscale('log')
plt.xlim(n[0], n[-1])
plt.xticks(n, n)
plt.yticks()
plt.xlabel('Matrix size (n)')
plt.ylabel('Time (s)')
plt.draw()

plt.figure(figsize=(5,4))
if cuda.any(): plt.plot(n[0:len(cuda)], cuda[:,1], marker='x', linestyle='-', linewidth=2, markersize=6, color='C0', label=r"CUDA")#($f_{child} = 2, n_{p} = 64$)")
if target.any(): plt.plot(n[0:len(target)], target[:,1], marker='v', linestyle='-', linewidth=2, markersize=6, color='C1', label=r"OpenMP Offloading")#($f_{child} = 2, n_{p} = 64$)")
if threads.any(): plt.plot(n[0:len(threads)], threads[:,1], marker='o', linestyle='-', linewidth=2, markersize=6, color='C2', label=r"OpenMP Threading")#($f_{child} = 2, n_{p} = 64$)")
if order.any(): plt.plot(n[0:len(order)], order[:,1], marker='>', linestyle='-', linewidth=2, markersize=6, color='C4', label=r"Row-Major")#($f_{child} = 2, n_{p} = 64$)")
if naive.any(): plt.plot(n[0:len(naive)], naive[:,1], marker='<', linestyle='-', linewidth=2, markersize=6, color='C5', label=r"Column-Major")#($f_{child} = 2, n_{p} = 64$)")
plt.legend(loc='upper left')
plt.xscale('log')
#plt.yscale('log')
plt.ylim(0, 16)
plt.xlim(n[0], n[-1])
plt.xticks(n, n)
plt.yticks()
plt.xlabel('Matrix size (n)')
plt.ylabel('Time (s)')
plt.draw()

plt.show()