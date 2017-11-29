import noise
import numpy as np
from PIL import Image

def perlin_noise(size):
    q = np.ndarray((size, size,size))
    for x in range(size):
        for y in range(size):
            for z in range(size):
                q[x,y,z] = (noise.pnoise3(x/size,y/size,z/size, octaves=16, persistence=0.5, lacunarity=2.0, repeatx=size, repeaty=size,repeatz=size,  base=0))
    print(q.max())
    print(q.min())
    return q + 0.5

