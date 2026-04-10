import numpy as np
import matplotlib.pyplot as plt
import socket
from skimage import measure, morphology


host = '84.237.21.36'
port = 5152


def get_points(image):
    threshold = image > image.mean() + image.std()
    cleaned = morphology.remove_small_objects(threshold, max_size=5)
    labeled = measure.label(cleaned)
    regions = measure.regionprops(labeled, intensity_image=image)
    regions = sorted(regions, key=lambda region: region.intensity_mean, reverse=True)
    pos1 = regions[0].centroid
    pos2 = regions[1].centroid
    return pos1, pos2

def recvall(sock, nbytes):
    data = bytearray()
    while len(data) < nbytes:
        packet = sock.recv(nbytes - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

plt.ion()
plt.figure()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((host, port))
    sock.send(b'124ras1')
    print(sock.recv(10))

    for i in range(10):
        sock.send(b'get')
        bts = recvall(sock, 40002)

        im = np.frombuffer(bts[2:], dtype='uint8').reshape(bts[0], bts[1])
        pos1, pos2 = get_points(im)

        dist = ((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2) ** 0.5
        sock.send(f'{round(dist, 1)}'.encode())
        print(f'{i+1}/10 dist={round(dist, 1)}, response="{sock.recv(10)}"')

        plt.clf()
        plt.imshow(im, cmap='gray')
        plt.show()
        plt.pause(2)

        sock.send(b'beat')
        beat = sock.recv(10)