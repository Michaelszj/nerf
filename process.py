import numpy as np
import taichi as ti
import taichi.math as tm
import time
import os
import cv2
from tqdm import tqdm
from voxel import *
path = 'Lego/pose/0_00'
appendix = '.txt'
pixels = None

image = None


def load_data():
    dataset = 'Lego/rgb/'
    images = []
    print("Loading {}".format(dataset))

    # Iterate through each image in our folder
    for file in tqdm(os.listdir(dataset)):
        if file[0] != '0':
            break

        # Get the path name of the image
        img_path = dataset+file

        # Open and resize the img
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (800, 800))
        image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_CLOCKWISE)

        # Append the image and its corresponding label to the output
        image = np.array(image, dtype='float32')
        image /= 255.0
        data = ti.field(ti.f32, shape=image.shape)
        data.from_numpy(image)
        images.append(data)

    return images


def load_mat():
    dataset = 'Lego/pose/'
    mats = []

    print("Loading {}".format(dataset))

    # Iterate through each image in our folder
    for file in tqdm(os.listdir(dataset)):
        if file[0] != '0':
            break

        # Get the path name of the image
        pose_path = dataset+file
        pose = np.loadtxt(pose_path, dtype=np.float32)

        # Append the image and its corresponding label to the output

        mats.append(pose)
    mats = np.array(mats)
    data = ti.Matrix.field(4, 4, ti.f32, shape=mats.shape[0])
    data.from_numpy(mats)

    return data


@ti.func
def mul(vec: vec3, mat: mat4):
    return vec3(mat[0, 0]*vec[0]+mat[0, 1]*vec[1]+mat[0, 2]*vec[2]+mat[0, 3],
                mat[1, 0]*vec[1]+mat[1, 1]*vec[1]+mat[1, 2]*vec[2]+mat[1, 3],
                mat[2, 0]*vec[0]+mat[2, 1]*vec[1]+mat[2, 2]*vec[2]+mat[2, 3])


@ti.kernel
def calRays(index: int, rays: ti.template(), c2w: ti.template(), origins: ti.template()):
    origins[index] = mul(vec3(0.0, 0.0, 0.0), c2w[index])
    for i, j in ti.ndrange(800, 800):
        rays[index, i, j] = tm.normalize(
            mul(vec3((ti.cast(i, ti.f32)-399.5)/400.0, (ti.cast(j, ti.f32)-399.5)/400.0, 1.0), c2w[index])-origins[index])


if __name__ == '__main__':
    pixels = load_data()
    c2w = load_mat()
    rays = ti.field(ti.f32, shape=(len(pixels), pixels[0].shape[0], pixels[0].shape[1], 3))
    origins = ti.field(ti.f32, shape=(len(pixels), 3))
    print(len(pixels), c2w.shape)
