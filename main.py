import PIL
import cProfile, pstats, io
from PIL import Image
import numpy as np
import perlin


# def wraparound(d):
#     d[d > 0.5] = (1 - d)[d > 0.5]
#     return d

#faster wraparound, 4 time faster
def wraparound(d):
    print(d.dtype)
    return 0.5-np.abs(d-0.5)

def wrap_distance_matrix(m,n):
    d = np.abs(m[:, np.newaxis] - n)
    d = wraparound(d)
    d=d*d
    return np.sqrt(d.sum(axis=2))

def worley_noise(size,  seed_points):
    q = np.moveaxis(np.mgrid[:size, :size, :size].astype(dtype=np.float32), 0, -1)
    q = q.reshape(size**3,3) / size
    q = wrap_distance_matrix(q,seed_points).min(axis=1,keepdims=True)
    return q.reshape((size, size, size))

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
    q = np.zeros((size, size, size),dtype=np.float32)
    max_point_count = point_count*np.power(2,octave-1)
    seed = np.random.rand(max_point_count*3).astype(dtype=np.float32).reshape(max_point_count,3)
    v = 1
    for i in range(octave):
        noise = worley_noise(size, seed[:point_count]) * v

        q += noise
        point_count *=2
        v*=0.5
    return q

def paste_array(big, small, top_left,size):
    # print("paste at:",top_left,size)
    big[top_left[0]:top_left[0]+size[0],top_left[1]:top_left[1]+size[1]] = small

def conv_3dto2d(a3dimgarray,step):
    z_dim =  a3dimgarray.shape[2]
    num_slices = z_dim//step
    slice_per_row = np.ceil(np.sqrt(num_slices))
    dim = a3dimgarray.shape[0]
    w = int(slice_per_row)
    h = int(np.ceil(num_slices/slice_per_row))
    img = np.ndarray((dim*w,dim*h))
    img = np.zeros_like(img)
    for i in range(0,num_slices,step):
        # print("taking slice:",i)
        paste_array(img,a3dimgarray[i],((i%w)*dim, (i//w)*dim),(dim,dim))
    return img


pr = cProfile.Profile()
pr.enable()

worley = fractal_worley(128,3,32)

pr.disable()
pr.create_stats()
pr.print_stats()

worley = normalize_and_smooth_signal(worley)
#pn = perlin.perlin_noise(128)
#worley *= pn

worley *=255
img = Image.fromarray(conv_3dto2d(worley,1))
img.show()

