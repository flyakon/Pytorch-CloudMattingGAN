import numpy as np
import cv2
import time
import torch

def generate_perlin_noise_2d(shape, res, tileable=(False, False)):
    '''
    generate perlin noise
    :param shape:
    :param res:
    :param tileable:
    :return:
    '''
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5, lacunarity=2, tileable=(False, False)):
    '''
    fbm
    :param shape:
    :param res:
    :param octaves:
    :param persistence:
    :param lacunarity:
    :param tileable:
    :return:
    '''
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for i in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]), tileable)
        if i>=6:
            lacunarity=1.5
        frequency *= lacunarity
        frequency=int(frequency)
        amplitude *= persistence
    return noise

def generate_cloud(img_size,res,octs,persistence=0.5, lacunarity=2,seed=None):
    t = time.time()
    if seed is None:
        seed=int(round(t*1000))%(2**32)
    np.random.seed(seed)
    noise = generate_fractal_noise_2d(img_size, res, octs,persistence, lacunarity)
    cv2.normalize(noise, noise, 0, 255, cv2.NORM_MINMAX)
    cloud_img = noise.astype(np.uint8)
    return cloud_img

def generate_alpha(cloud_img):
    dtype=None
    if isinstance(cloud_img,torch.Tensor):
        dtype=torch.Tensor
        cloud_img=cloud_img.numpy()
    alpha = np.exp(-cloud_img)
    cv2.normalize(alpha, alpha, 0., 1., cv2.NORM_MINMAX)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    alpha=cv2.blur(alpha,(5,5))
    if dtype==torch.Tensor:
        alpha=torch.from_numpy(alpha)
    return  alpha



