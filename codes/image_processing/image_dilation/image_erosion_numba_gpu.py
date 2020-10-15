import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
from skimage import io

image = io.imread("letter_image.png")
image_shape = image.shape[0]

image_eroded = np.zeros_like(image)

@cuda.jit
def erode(image, result):
    image_height, image_width = image.shape

    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if 1 < row < image_height - 1:
        if 1 < col < image_width - 1:
            extreme_value = 1
            for i in range(-1, 2):
                for j in range(-1, 2):
                    pixel_value = image[row + i, col + j]
                    extreme_value = min(pixel_value, extreme_value)
            result[row, col] = extreme_value

threads_per_block = (16, 16)
blocks_per_grid_x = int(np.ceil(image_shape / threads_per_block[0]))
blocks_per_grid_y = int(np.ceil(image_shape / threads_per_block[1]))
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# call erosion kernel
erode[blocks_per_grid, threads_per_block](image, image_eroded)

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
