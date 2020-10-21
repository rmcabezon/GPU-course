#%% md

### Scikit-Image version

#%%
# define image path
import functools
import time

image_path = "/home/micha/Documents/01_work/04_teaching/01_gpgpu_course_scicore/gpu_course/codes/image_processing/sobel_v_convolution/photographer.png"

#%% md

# Define some support functions for loading/tiling the image and showing it and its filtered version.

#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, io

def read_image(image_path, tiles_per_dimension=1):
    # read image and normalize
    image = io.imread(image_path).astype(np.float32)
    image -= np.min(image)
    image /= np.max(image)
    image = np.tile(image, (tiles_per_dimension, tiles_per_dimension))
    return image

def keep_valid_image(filtered_image, filter):
    # ignore edges, which are not a valid result
    filter_height, filter_width = filter.shape
    filter_height_halved = filter_height // 2
    filter_width_halved = filter_width // 2
    return filtered_image[filter_height_halved:-filter_height_halved,
                         filter_width_halved:-filter_width_halved]


def show_image(image_orig, image_filtered):
    # show images
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))
    axes[0].imshow(image_orig, cmap=plt.cm.gray)
    axes[0].set_title('Original')
    axes[1].imshow(image_filtered, cmap=plt.cm.gray)
    axes[1].set_title('Processed')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

from contextlib import contextmanager
@contextmanager
def timeit_context(name, repeats=1):
    startTime = time.time()
    for ind in range(repeats):
        yield
    elapsedTime = time.time() - startTime
    print('{} finished in {} ms'.format(name, elapsedTime * 1000./repeats))

#%% md

### Scikit-Image version

#%%
image = read_image(image_path)

# filter vertical edges with sobel_v
edges_sobel_v = filters.sobel_v(image)

# print min- and max-values for comparison with reference implementation
print("min pixel value: " + str(np.min(edges_sobel_v)))
print("max pixel value: " + str(np.max(edges_sobel_v)))

show_image(image, edges_sobel_v)

#%% md

### CUDA implementation

#%%

# Let's write our own CPU version of the convolution function using Numba.

#%%
# from numba import jit, autojit, prange
from numba import jit

# kernel definition
@jit(nopython=True, parallel=True)
def filter2d_cpu(image, filt, result):
    image_height, image_width = image.shape
    filter_height, filter_width = filt.shape
    filter_height_halved = filter_height // 2
    filter_width_halved = filter_width // 2

    for row in range(0, image_height):
        for col in range(0, image_width):
            if (row > filter_height_halved and row < image_height - filter_height_halved):
                if (col > filter_width_halved and col < image_width - filter_width_halved):
                    sum = 0.0
                    for conv_index_y in range(-filter_height_halved, filter_height_halved + 1):
                        for conv_index_x in range(-filter_width_halved, filter_width_halved + 1):
                            kernelCoord = filter_height_halved + conv_index_y, filter_width_halved + conv_index_x
                            imageCoord = row + conv_index_y, col + conv_index_x
                            sum += filt[kernelCoord] * image[imageCoord]
                    result[row, col] = sum

#%%
import numpy as np

# read image and define output image
image = read_image(image_path, 1)
edges_sobel_v = np.zeros_like(image)

# Define sobel filter for vertical edges
sobel_v_filter = (1. / 4) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

# filter image
filter2d_cpu(image, sobel_v_filter, edges_sobel_v)

# strip invalid image regions and show
edges_sobel_v = keep_valid_image(edges_sobel_v, sobel_v_filter)
show_image(image, edges_sobel_v)

print("min pixel value: " + str(np.min(edges_sobel_v)))
print("max pixel value: " + str(np.max(edges_sobel_v)))

#%% md

# Same result. Great. Now let's rewrite the convolution functions using a naive CUDA implementation.

#%%
from numba import cuda

# kernel definition
@cuda.jit
def filter2d_gpu(image, filt, result):
    image_height, image_width = image.shape
    filter_height, filter_width = filt.shape
    filter_height_halved = filter_height // 2
    filter_width_halved = filter_width // 2

    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if (row > filter_height_halved and row < image_height - filter_height_halved):
        if (col > filter_width_halved and col < image_width - filter_width_halved):
            sum = 0.0
            for conv_index_y in range(-filter_height_halved, filter_height_halved + 1):
                for conv_index_x in range(-filter_width_halved, filter_width_halved + 1):
                    kernelCoord = filter_height_halved + conv_index_y, filter_width_halved + conv_index_x
                    imageCoord = row + conv_index_y, col + conv_index_x
                    sum += filt[kernelCoord] * image[imageCoord]
            result[row, col] = sum

#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
from skimage import io

# read image and define output image
image = read_image(image_path, 1)
edges_sobel_v = np.zeros_like(image)

# Define sobel filter for vertical edges
sobel_v_filter = (1. / 4) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

filter_height, filter_width = sobel_v_filter.shape
filter_height_halved = filter_height // 2
filter_width_halved = filter_width // 2

threads_per_block = (16, 16)
image_shape = image.shape[0]
blocks_per_grid_x = int(np.ceil(image_shape / threads_per_block[0]))
blocks_per_grid_y = int(np.ceil(image_shape / threads_per_block[1]))
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# filter vertical edges with sobel_v_filter

# resCPU = %timeit -n 2 -r 10 -o filter2d_cpu(image, sobel_v_filter, edges_sobel_v)
with timeit_context('CPU performance:'):
    filter2d_cpu(image, sobel_v_filter, edges_sobel_v)

# resGPU = %timeit -n 2 -r 10 -o filter2d_gpu[blocks_per_grid, threads_per_block](image, sobel_v_filter, edges_sobel_v)
with timeit_context('GPU performance:'):
    filter2d_gpu[blocks_per_grid, threads_per_block](image, sobel_v_filter, edges_sobel_v)

# strip invalid image regions and show
edges_sobel_v = keep_valid_image(edges_sobel_v, sobel_v_filter)
show_image(image, edges_sobel_v)

print("min pixel value: " + str(np.min(edges_sobel_v)))
print("max pixel value: " + str(np.max(edges_sobel_v)))

#%% md

# GPU and CPU perform with same performance. But what happens for larger images? Let's try with 10x10 tiling of the image ...

#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
from skimage import io

# read image and define output image
image = read_image(image_path, 10)
edges_sobel_v = np.zeros_like(image)

# Define sobel filter for vertical edges
sobel_v_filter = (1. / 4) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

filter_height, filter_width = sobel_v_filter.shape
filter_height_halved = filter_height // 2
filter_width_halved = filter_width // 2

threads_per_block = (16, 16)
image_shape = image.shape[0]
blocks_per_grid_x = int(np.ceil(image_shape / threads_per_block[0]))
blocks_per_grid_y = int(np.ceil(image_shape / threads_per_block[1]))
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# filter vertical edges with sobel_v_filter

# resCPU = %timeit -n 2 -r 10 -o filter2d_cpu(image, sobel_v_filter, edges_sobel_v)
with timeit_context('CPU performance:'):
    filter2d_cpu(image, sobel_v_filter, edges_sobel_v)

# resGPU = %timeit -n 2 -r 10 -o filter2d_gpu[blocks_per_grid, threads_per_block](image, sobel_v_filter, edges_sobel_v)
with timeit_context('GPU performance:'):
    filter2d_gpu[blocks_per_grid, threads_per_block](image, sobel_v_filter, edges_sobel_v)
    cuda.synchronize()

show_image(image, edges_sobel_v)

#%% md

# There we go: GPU is significantly faster. The missing speed-up in the first example is due to memory transfer to the GPU.
# Let's measure the time spent on data transfer and computation on the GPU.

#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
from skimage import io

# read image and define output image
image = read_image(image_path, 13)
edges_sobel_v = np.zeros_like(image)

# Define sobel filter for vertical edges
sobel_v_filter = (1. / 4) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

filter_height, filter_width = sobel_v_filter.shape
filter_height_halved = filter_height // 2
filter_width_halved = filter_width // 2

threads_per_block = (16, 16)
image_shape = image.shape[0]
blocks_per_grid_x = int(np.ceil(image_shape / threads_per_block[0]))
blocks_per_grid_y = int(np.ceil(image_shape / threads_per_block[1]))
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

with timeit_context('GPU processing time with transfer'):
    filter2d_gpu[blocks_per_grid, threads_per_block](image, sobel_v_filter, edges_sobel_v)
    cuda.synchronize()

print("")

# copy to arrays to device memory before processing
stream = cuda.stream()
with timeit_context('Transfer to GPU'):
    image_on_device = cuda.to_device(image, stream=stream)
    filter_on_device = cuda.to_device(sobel_v_filter, stream=stream)
    filtered_image_on_device = cuda.to_device(edges_sobel_v, stream=stream)
    cuda.synchronize()

with timeit_context('GPU processing time without transfer'):
    filter2d_gpu[blocks_per_grid, threads_per_block](image_on_device, filter_on_device, filtered_image_on_device)
    cuda.synchronize()

# get arrays from device memory before processing
with timeit_context('Transfer from GPU'):
    image_on_host = image_on_device.copy_to_host()
    filter_on_host = filter_on_device.copy_to_host()
    filtered_image_on_host = filtered_image_on_device.copy_to_host()
    cuda.synchronize()

show_image(image, filtered_image_on_host)

#%% md
# So we spend much time with transfering the data to the GPU.
# Conclusions are:
# (1) A workload must be sufficiently large in order for the speed-up on GPU to justify the time spent on data transfer.
# (2) For computations consisting of multiple CUDA kernels, we should try to keep processed data in GPU memory as long
#     possible and avoid unnecessary data-transfer.
# (3) You can try to transfer data to the GPU asynchronously in parallel with other computation.(???)

