import noise
import numpy as np
from PIL import Image

def perlin_noise(size):
    q = np.ndarray((size, size))
    for x in range(size):
        for y in range(size):
            q[x,y] = (noise.pnoise2(x/size,y/size,octaves=32, persistence=0.5, lacunarity=2.0, repeatx=512, repeaty=512, base=0))
    print(q.max())
    print(q.min())
    return q + 0.5

perlin_noise(512)