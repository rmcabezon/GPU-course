import numpy as np
import time
import math

from numpy.random import seed
from numpy.random import rand
from numba import njit,prange

@njit
def compute_3d_k(n):
    b0 = 2.7012593e-2;
    b1 = 2.0410827e-2;
    b2 = 3.7451957e-3;
    b3 = 4.7013839e-2;

    return b0 + b1 * math.sqrt(n) + b2 * n + b3 * math.sqrt(n * n * n);

# This is the SPH Kernel
@njit
def wharmonic(v):
    if (v == 0.0):
        return 1.0;
    Pv = (math.pi / 2.0) * v;
    return math.sin(Pv) / Pv;

# Compute density here
@cuda.jit
def compute_density_cuda(n, ngmax, neighbors, neighborsCount, x, y, z, h, m, ro, K, offset=0):
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    i = offset + tx + ty * bw
    
    if i < n:
        nn = neighborsCount[i];

        roloc = 0.0;

        for pj in prange(0, nn):
            j = neighbors[(i-offset) * ngmax + pj];

            xx = x[i] - x[j];
            yy = y[i] - y[j];
            zz = z[i] - z[j];

            dist = math.sqrt(xx * xx + yy * yy + zz * zz);

            # SPH Kernel
            vloc = wharmonic(dist / h[i]);

            w = K * vloc * vloc * vloc * vloc * vloc * vloc;
            value = w / (h[i] * h[i] * h[i]);

            roloc = roloc + value * m[j];

        ro[i] = roloc + m[i] * K / (h[i] * h[i] * h[i]);

# Reads the input file
# Calls compute_density
# Write the result in out.txt
f = open('/scicore/home/scicore/GROUP/gpu_course/pdata')

if f.closed == True:
    print("Error opening file pdata")

n = int(np.fromfile(f, dtype=np.int64, count=1))
ngmax = int(np.fromfile(f, dtype=np.int64, count=1))

x = np.fromfile(f, dtype=np.double, count=n)
y = np.fromfile(f, dtype=np.double, count=n)
z = np.fromfile(f, dtype=np.double, count=n)
h = np.fromfile(f, dtype=np.double, count=n)
m = np.fromfile(f, dtype=np.double, count=n)
neighborsCount = np.fromfile(f, dtype=np.int32, count=n)
neighbors = np.fromfile(f, dtype=np.int32, count=n*ngmax)

f.close()

ro = np.empty([n], dtype=np.double)

start = time.time()
compute_density(n, ngmax, neighbors, neighborsCount, x, y, z, h, m, ro);
end = time.time()

print('Elapsed time: ', end - start)

start = time.time()
compute_density(n, ngmax, neighbors, neighborsCount, x, y, z, h, m, ro);
end = time.time()

print('Elapsed time: ', end - start)

f = open('density.txt', 'w')

for i in prange(0, n):
    f.write(str(x[i]) + " " + str(y[i]) + " " + str(z[i]) + " " + str(ro[i]) + "\n")

f.close()
