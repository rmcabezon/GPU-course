import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
from skimage import io

# read image and normalize
image = io.imread('photographer.png').astype(np.float32)
image -= np.min(image)
image /= np.max(image)

# Sobel filter for vertical edges
tmp = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobel_v_filter = (1. / 4) * np.array(tmp, dtype=np.float32)

filter_height, filter_width = sobel_v_filter.shape
filter_height_halved = filter_height // 2
filter_width_halved = filter_width // 2

# get shape of input image, allocate memory for output to which result can be copied to
shape = image.T.shape
edges_sobel_v = np.empty_like(image)

# setup OpenCL
platforms = cl.get_platforms()  # a platform corresponds to a driver (e.g. AMD)
platform = platforms[0]  # take first platform
devices = platform.get_devices(cl.device_type.GPU)  # get GPU devices of selected platform
device = devices[0]  # take first GPU
context = cl.Context([device])  # put selected GPU into context object
queue = cl.CommandQueue(context, device)  # create command queue for selected GPU and context

# create image buffers which hold images for OpenCL
image_buffer = cl.image_from_array(context, ary=image, mode="r", num_channels=1)
sobel_v_filter_buffer = cl.image_from_array(context, ary=sobel_v_filter, mode="r", num_channels=1)
edges_sobel_v_buffer = cl.image_from_array(context, ary=edges_sobel_v, mode="w", num_channels=1)

# load and compile OpenCL program
program = cl.Program(context, open(
    'convolution_kernel_code.cl').read()).build()

# filter vertical edges sobel_v_filter
program.custom_convolution_2d(queue, shape, None, image_buffer, sobel_v_filter_buffer, edges_sobel_v_buffer)

# copy back output buffer
cl.enqueue_copy(queue, edges_sobel_v, edges_sobel_v_buffer, origin=(0, 0), region=shape,
                is_blocking=True)  # block until finished copying result image back from GPU to CPU

# ignore edges, which are not a valid convolution result
edges_sobel_v = edges_sobel_v[filter_height_halved:-filter_height_halved,
                              filter_width_halved:-filter_width_halved]

# print min- and max-values for comparison with reference implementation
print("min: " + str(np.min(edges_sobel_v)))
print("max: " + str(np.max(edges_sobel_v)))

# show images
fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))
axes[0].imshow(image, cmap=plt.cm.gray)
axes[0].set_title('Image')
axes[1].imshow(edges_sobel_v, cmap=plt.cm.gray)
axes[1].set_title('Vertical edges')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
