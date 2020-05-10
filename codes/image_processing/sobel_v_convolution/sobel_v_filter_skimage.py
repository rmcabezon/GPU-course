import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, io

# read image and normalize
image = io.imread("photographer.png").astype(np.float32)
image -= np.min(image)
image /= np.max(image)

# filter vertical edges with sobel_v
edges_sobel_v = filters.sobel_v(image)

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
