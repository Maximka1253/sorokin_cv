import numpy as np
from skimage.color import rgb2hsv
from skimage.io import imread
from skimage.measure import label, regionprops


img = imread("balls_and_rects.png")

def find_near_shade(shade, shades):
    for s in shades:
        if abs(s - shade) < 0.05:
            return s
    return shade


h = rgb2hsv(img)[:, :, 0]

fg = np.any(img > 0, axis=2)
lab = label(fg)

shades = []
all_stats = {}
rect_stats = {}
circle_stats = {}

for reg in regionprops(lab):
    reg_mask = lab == reg.label
    avg = float(np.mean(h[reg_mask]))

    shade = find_near_shade(avg, shades)
    if shade == avg:
        shades.append(avg)

    all_stats[shade] = all_stats.get(shade, 0) + 1

    if reg.extent > 0.95:
        rect_stats[shade] = rect_stats.get(shade, 0) + 1
    else:
        circle_stats[shade] = circle_stats.get(shade, 0) + 1

total = sum(all_stats.values())
print(f"Всего фигур: {total}")

print("\nВсе фигуры по оттенкам:")
for shade, count in sorted(all_stats.items()):
    print(f"Оттенок {shade:.3f}: {count}")

print("\nПрямоугольники по оттенкам:")
for shade, count in sorted(rect_stats.items()):
    print(f"Оттенок {shade:.3f}: {count}")

print("\nКруги по оттенкам:")
for shade, count in sorted(circle_stats.items()):
    print(f"Оттенок {shade:.3f}: {count}")
