import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
from skimage import io

# read image
image = io.imread("letter_image.png")

# get shape of input image, allocate memory for output to which result can be copied to
shape = image.T.shape
image_eroded = np.empty_like(image)

# setup OpenCL context and queue
platforms = cl.get_platforms()  # a platform corresponds to a driver (e.g. AMD)
platform = platforms[0]  # take first platform
devices = platform.get_devices(cl.device_type.GPU)  # get GPU devices of selected platform
device = devices[0]  # take first GPU
context = cl.Context([device])  # put selected GPU into context object
queue = cl.CommandQueue(context, device)  # create command queue for selected GPU and context

# create image buffers which hold images for OpenCL
imgInBuf = cl.Image(context, cl.mem_flags.READ_ONLY,
                    cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8),
                    shape=shape)  # holds a gray-valued image of given shape
image_eroded_buffer = cl.Image(context, cl.mem_flags.WRITE_ONLY,
                               cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8),
                               shape=shape)  # placeholder for gray-valued image of given shape

# load and compile OpenCL program
program = cl.Program(context, open('erosion_kernel.cl').read()).build()

# from OpenCL program, get kernel object and set arguments (input image, operation type, output image)
kernel = cl.Kernel(program, 'morphOpKernel')  # name of function according to kernel.py
kernel.set_arg(0, imgInBuf)  # input image buffer
kernel.set_arg(1, image_eroded_buffer)  # output image buffer

# copy image to device, execute kernel, copy data back
cl.enqueue_copy(queue, imgInBuf, image, origin=(0, 0), region=shape,
                is_blocking=False)  # copy image from CPU to GPU
cl.enqueue_nd_range_kernel(queue, kernel, shape,
                           None)  # execute kernel, work is distributed across shape[0]*shape[1] work-items (one work-item per pixel of the image)
cl.enqueue_copy(queue, image_eroded, image_eroded_buffer, origin=(0, 0), region=shape,
                is_blocking=True)  # wait until finished copying resulting image back from GPU to CPU

# keep only valid range
image_eroded = image_eroded[1:-1, 1:-1]

# show images
fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))
axes[0].imshow(image, cmap=plt.cm.gray)
axes[0].set_title('Image')
axes[1].imshow(image_eroded, cmap=plt.cm.gray)
axes[1].set_title('Image eroded')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
