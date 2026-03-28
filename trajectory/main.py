import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from pathlib import Path

folder = Path("/home/s/Рабочий стол/computer-vision git/trajectory/out/")
files = sorted(folder.glob("h_*.npy"), key=lambda p: int(p.stem.split('_')[1]))

first_regions = sorted(regionprops(label(np.load(files[0]))), key=lambda r: r.centroid[1])
trajectories = [[(r.centroid[1], r.centroid[0])] for r in first_regions]

for f in files[1:]:
    centroids = [r.centroid for r in regionprops(label(np.load(f)))]
    
    for traj in trajectories:
        last_y, last_x = traj[-1][1], traj[-1][0]
        best = min(centroids, key=lambda c: (c[0]-last_y)**2 + (c[1]-last_x)**2)
        traj.append((best[1], best[0]))

plt.figure(figsize=(10, 6))
for i, points in enumerate(trajectories):
    x, y = zip(*points)
    plt.plot(x, y, '-o', markersize=3, label=f'Объект {i+1}')

plt.gca().invert_yaxis()
plt.legend()
plt.show()