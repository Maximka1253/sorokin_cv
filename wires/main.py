import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import (opening,
                                dilation,
                                closing,
                                erosion)

image = np.load("/home/s/Рабочий стол/wires/wires5.npy")
struct = np.ones((3,1))
processed = opening(image, footprint=struct)
labeled = label(image)
print(f"{labeled.max()}")

for n in range(1, labeled.max()+1):
    mask = (labeled==n)
    part = label(opening(mask, footprint=struct)).max()
    print(f"Wire = {n}, parts = {part}")

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(processed)
plt.show()