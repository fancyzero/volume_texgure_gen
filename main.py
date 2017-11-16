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


def nearestn(pt_index, distance_matrix):
    return distance_matrix[pt_index][0]


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
        q[i]=nearestn(i,distances)
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

def fractal_worley(size, octave):
    base_point_count = int(size //8)
    q = np.zeros((size, size))
    max_point_count = base_point_count*np.power(2,octave-1)
    seed = np.random.rand(max_point_count*2).reshape(max_point_count,2)
    for i in range(octave):
        seed_cur = seed[:base_point_count*np.power(2,i)]
        q += worley_noise(size, seed_cur) * (1.0 / (i + 1))

    # normalize everyting out
    q = q / q.max()
    q = 1 - q
    return q

worley = fractal_worley(512,1)

#pn = perlin.perlin_noise(128)

#worley *= pn
worley /= worley.max()
worley *=255
Image.fromarray(worley).show()