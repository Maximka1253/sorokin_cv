import numpy as np
from scipy.ndimage import binary_erosion, label

data = np.load("stars.npy")
binary = data > 0

plus = np.array([[0,0,1,0,0],
                 [0,0,1,0,0],
                 [1,1,1,1,1],
                 [0,0,1,0,0],
                 [0,0,1,0,0,]])

cross = np.array([[1,0,0,0,1],
                  [0,1,0,1,0],
                  [0,0,1,0,0],
                  [0,1,0,1,0],
                  [1,0,0,0,1]])

crosses = binary_erosion(binary, structure=cross)
pluses = binary_erosion(binary, structure=plus)

_, plus_count = label(pluses)
_, cross_count = label(crosses)

print(f"Плюсы: {plus_count}")
print(f"Крестики: {cross_count}")
