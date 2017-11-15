import PIL
from PIL import Image
import numpy as np


def distance_matrix(m,n):
    d = (np.abs(m[:, np.newaxis] - n))
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


def nearestn(pt, points):
    distances=[]
    for p in points:
        distances.append(distance(p,pt))

    distances = np.sort(distances)
    return distances[0]


def worley_noise(dim, n):
    q = np.ndarray((dim,dim))
    t = np.random.rand(n*2)
    t= t.reshape((n,2))

    for x in range(dim):
        for y in range(dim):
            fx = x/float(dim)
            fy = y/float(dim)
            q[x,y] = (fx,fy)

    print (q)
    q = q/q.max()
    q=1-q
    q*=q
    q = 255*q
    return q



d = 128
q = worley_noise(d,60)
img = Image.fromarray(q)
img2 = Image.new("RGB",(d*3,d*3))
img2.paste(img,box=(0,0))
img2.paste(img,box=(d,0))
img2.paste(img,box=(d*2,0))
img2.paste(img,box=(0,d))
img2.paste(img,box=(d,d))
img2.paste(img,box=(d*2,d))
img2.paste(img,box=(0,d*2))
img2.paste(img,box=(d,d*2))
img2.paste(img,box=(d*2,d*2))

img2.show()



