import PIL
from PIL import Image
import numpy as np
import perlin

def wrap_distance_matrix(m,n):
    d = (np.abs(m[:, np.newaxis] - n))
    d[d>0.5] = (1-d)[d > 0.5]
    d=d*d
    return np.sqrt(d.sum(axis=2))

def distance(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    m = np.abs(p1-p2)
    if m[0] > 0.5:
        m[0] = 1 - m[0]
    if m[1] > 0.5:
        m[1] = 1 - m[1]
    return np.linalg.norm(m)

def worley_noise(size, seed_points):
    q = np.ndarray((size * size, 2))

    for x in range(size):
        for y in range(size):
            fx = x/float(size)
            fy = y/float(size)
            q[x * size + y] = (fx, fy)


    distances = wrap_distance_matrix(q,seed_points)
    distances.sort(axis=1)

    q = np.ndarray((size * size))
    for i in range(size*size):
        q[i]=distances[i][0]
    return q.reshape((size, size))



def show_image_as_tiled(q):
    img = Image.fromarray(q)
    img2 = Image.new("RGB", (img.width * 3, img.height * 3))
    img2.paste(img,box=(0,0))
    img2.paste(img, box=(img.width, 0))
    img2.paste(img, box=(img.width * 2, 0))
    img2.paste(img, box=(0, img.height))
    img2.paste(img, box=(img.width, img.height))
    img2.paste(img, box=(img.width * 2, img.height))
    img2.paste(img, box=(0, img.height * 2))
    img2.paste(img, box=(img.width, img.height * 2))
    img2.paste(img, box=(img.width * 2, img.height * 2))

    img2.show()

def normalize_and_smooth_signal(image):
    image = image / image.max()
    image = 1 - np.clip(image * image * (3 - 2 * image), 0, 1)
    return image

def fractal_worley(size, octave, point_count):
    q = np.zeros((size, size))
    max_point_count = point_count*np.power(2,octave-1)
    seed = np.random.rand(max_point_count*2).reshape(max_point_count,2)
    v = 1
    for i in range(octave):
        noise = worley_noise(size, seed[:point_count]) * v

        q += noise
        point_count *=2
        v*=1
    return q

worley = fractal_worley(128,6,32)
worley = normalize_and_smooth_signal(worley)
#pn = perlin.perlin_noise(128)
#worley *= pn

worley *=255
Image.fromarray(worley).show()