import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

save_path = Path(__file__).parent

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1

def count_lines(region):
    shape = region.image.shape
    image = region.image
    vlines = (np.sum(image, 0)/ shape[0] == 1).sum()
    hlines = (np.sum(image, 1)/ shape[1] == 1).sum()
    return vlines, hlines

def symmetry(region, transpose=False):
    image = region.image
    if transpose:
        image = image.T
    shape = image.shape
    top = image[:shape[0]//2]
    if shape[0] % 2 != 0:
        bottom = image[shape[0]//2+1:]
    else:
        bottom = image[shape[0]//2:]
    bottom = bottom[::-1]
    result = bottom==top
    return result.sum() / result.size

def classificator(region):
    holes = count_holes(region)
    if holes == 2: #B, 8
        v, _ = count_lines(region)
        v /= region.image.shape[1]
        sym = symmetry(region, transpose = True)
        if v > 0.2 and sym < 0.9:
            return "B"
        else:
            return "8"
        pass
    elif holes == 1: #A, O
        eccentricity = region.eccentricity
        v_sym = symmetry(region)
        h_sym = symmetry(region, transpose = True)
        if v_sym > 0.98 and (eccentricity > 0.7 or eccentricity < 0.6):
            return 'D'
        if v_sym < 0.6 and h_sym > 0.7:
            return 'A'
        if h_sym > 0.8 and v_sym > 0.8:
            return 'O'
        else:
            return 'P'
    elif holes == 0: #1,W,X,*,-,/
        extent = region.extent
        v, _ = count_lines(region)
        if region.image.sum() / region.image.size > 0.95:
            return "-"
        shape = region.image.shape
        aspect = np.min(shape) / np.max(shape)
        if aspect > 0.9:
            return "*"
        v_asym = symmetry(region)
        h_asym = symmetry(region, transpose=True)
        if v_asym > 0.84 and h_asym > 0.84 and v < 0.1:
            return "X"
        elif v > 1 and v_asym < 1:
            return "1"
        elif h_asym > 0.7:
            return "W"
        elif v == 0:
            return "/"
    return "?"

image = imread("symbols.png")[:, :, :-1]
abinary = image.mean(2) > 0
alabeled = label(abinary)
print(np.max(alabeled))
aprops = regionprops(alabeled)
result = {}
image_path = save_path / "out"
image_path.mkdir(exist_ok=True)

# plt.ion()
plt.figure(figsize=(5, 7))
for region in aprops:
    symbol = classificator(region)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1
    plt.cla()
    plt.title(f"Class - {symbol}")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")

print(result)
plt.imshow(abinary)
plt.show()