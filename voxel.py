import taichi as ti
import taichi.math as tm
import time
import random
import numpy as np
import math
from gmath import *


window_shape = (640, 480)


@ti.data_oriented
class voxel_system:
    def __init__(self):
        self.content = ti.Vector.field(
            13, dtype=ti.float32, shape=(grid_l, grid_l, grid_l))
        self.grad = ti.Vector.field(
            13, dtype=ti.float32, shape=(grid_l, grid_l, grid_l))
        self.lr = ti.field(ti.f32, shape=())
        self.lr[None] = 0.001

    @ti.kernel
    def circle_init(self):
        for i, j, k in self.content:
            for l in ti.static(range(13)):
                if l == 2:
                    self.content[i, j, k][l] = (ti.random()-0.5)/4
                elif l == 12:
                    self.content[i, j, k][l] = ti.random()/40
                else:
                    self.content[i, j, k][l] = (ti.random()-0.5)/40
                if sum(tm.pow(vec3((i/half_l)-1, (j/half_l)-1, (k/half_l)-1), 2)) > 1.0:
                    self.content[i, j, k][l] /= 5

    @ti.kernel
    def random_init(self):
        for i, j, k in self.content:
            for l in ti.static(range(13)):
                if l == 12:
                    self.content[i, j, k][l] = ti.random()/4
                else:
                    self.content[i, j, k][l] = (ti.random()-0.5)/2

    @ti.kernel
    def zero_init(self):
        for i, j, k in self.content:
            for l in ti.static(range(13)):
                if l == 12:
                    self.content[i, j, k][l] = 0.9
                else:
                    self.content[i, j, k][l] = 0.

    @ti.kernel
    def update(self, cluster: bool):
        for i, j, k in self.content:
            if self.grad[i, j, k][0] >= 3.:
                if cluster == True:
                    self.content[i, j, k] = vec13(0.)
            else:
                # self.grad[i, j, k][12] = 0.
                self.content[i, j, k] -= self.grad[i, j, k]*self.lr[None]
            # if (self.grad[i, j, k][0] == 0.0):
            #     self.content[i, j, k][0] = 1.0
            for l in range(13):
                if l == 12:
                    # self.content[i, j, k][l] *= (1+self.lr[None])
                    if (self.content[i, j, k][l] < 0.):
                        self.content[i, j, k][l] = 0.
                    if (self.content[i, j, k][l] > 1.):
                        self.content[i, j, k][l] = 1.
                if l < 3:
                    if (self.content[i, j, k][l] < 0.):
                        self.content[i, j, k][l] = 0.

    @ti.kernel
    def zero_grad(self):
        temp = vec13(0.)
        for i, j, k in self.grad:
            self.grad[i, j, k] = temp


@ti.data_oriented
class Camera:
    def __init__(self, scene: voxel_system):
        # properties of a camera
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.target = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.target[None] = vec3(0.0, 0.0, 0.0)
        self.fov = ti.field(ti.f32, shape=())
        self.radius = ti.field(ti.f32, shape=())
        self.theta = ti.field(ti.f32, shape=())
        self.fai = ti.field(ti.f32, shape=())
        self.fov[None] = 13.0/180.0*tm.pi
        self.radius = 1000.0
        self.theta = tm.pi/6
        self.fai = tm.pi/4
        self.position[None] = vec3(self.radius*tm.cos(self.fai)*tm.sin(self.theta), self.radius *
                                   tm.sin(self.fai), -self.radius*tm.cos(self.fai)*tm.cos(self.theta))

        # belongings of a camera
        self.rays = ti.Vector.field(3, dtype=ti.f32, shape=window_shape)
        self.gui = ti.GUI('window', window_shape, fast_gui=True)

        # image buffer
        self.image = ti.Vector.field(3, dtype=ti.f32, shape=window_shape)

        # the voxel system that the camera belongs to
        self.scene = scene

        # gui control
        self.mouseDown = False
        self.closeWindow = False
        self.last_x = 0.0
        self.last_y = 0.0

    @ti.kernel
    # prepare for a ray marching process
    def cal_rays(self):
        direction = tm.normalize(self.target[None]-self.position[None])
        x_size = self.rays.shape[0]
        y_size = self.rays.shape[1]
        x_axis = tm.normalize(vec3(direction.z, 0.0, -direction.x))
        y_axis = tm.cross(direction, x_axis)
        g_length = tm.tan(self.fov[None])*(2.0/ti.cast(y_size, ti.f32))
        for i, j in self.rays:
            self.rays[i, j] = tm.normalize(
                direction+(i-x_size/2+0.5)*g_length*x_axis+(j-y_size/2+0.5)*g_length*y_axis)

    @ti.kernel
    # pixel computing
    def render(self):
        pos = self.position[None]
        for i, j in self.image:
            ray = self.rays[i, j]
            tmin, tmax = enterBox(pos, ray)
            dis = 0.0
            point = vec3(0.0)
            weight = 1.0
            color = vec3(0.0)
            if (tmin >= tmax) or (tmax < 0.0):
                self.image[i, j] = vec3(0.)  # ray*0.5
            else:
                dis = tmin+0.1
                while (dis < tmax):
                    point = pos+dis*ray
                    x, y, z, vol = toGridLerp(point)
                    paras = vec13(0.0)
                    for k in ti.static(range(8)):
                        paras += self.scene.content[x+k % 2, y+k//2 % 2, z+k//4 % 2]*vol[k]
                    tempColor, new_weight = SH_forward(paras, -ray, weight)
                    color += tempColor
                    dis += stride_length
                    weight = new_weight
                # color += ray*weight
                self.image[i, j] = color

    # process input from mouse and keyboard
    def processEvents(self):
        for e in self.gui.get_events():
            if e.key == ti.GUI.ESCAPE:
                self.gui.running = False
            elif e.key == ti.GUI.LMB and e.type == ti.GUI.PRESS:
                self.mouseDown = True
                self.last_x, self.last_y = self.gui.get_cursor_pos()
            elif e.key == ti.GUI.LMB and e.type == ti.GUI.RELEASE:
                self.mouseDown = False

        if self.mouseDown:
            x, y = self.gui.get_cursor_pos()
            self.theta -= (x-self.last_x)*3
            self.fai -= (y-self.last_y)*3
            self.theta -= math.floor(self.theta/(tm.pi*2))*tm.pi*2
            if (self.fai < -tm.pi):
                self.fai = -tm.pi
            if (self.fai > tm.pi):
                self.fai = tm.pi
            self.last_x = x
            self.last_y = y
            self.position[None] = vec3(self.radius*tm.cos(self.fai)*tm.sin(self.theta), self.radius *
                                       tm.sin(self.fai), -self.radius*tm.cos(self.fai)*tm.cos(self.theta))
            # print(x, y)

    # display the image buffer
    def display(self):
        self.gui.set_image(self.image)
        self.gui.show()


@ti.kernel
def renderBuffer(buffer: ti.template(), scene: ti.template(), origins: ti.template(), rays: ti.template(), index: int):
    pos = origins[index]
    for i, j in buffer:
        ray = rays[index, i, j]
        tmin, tmax = enterBox(pos, ray)
        dis = 0.0
        point = vec3(0.0)
        weight = 1.0
        color = vec3(0.0)
        if (tmin >= tmax) or (tmax < 0.0):
            buffer[i, j] = vec3(0.)
        else:
            if tmin < 0:
                tmin = 0.
            dis = tmin+0.1
            while (dis < tmax):
                point = pos+dis*ray
                x, y, z, vol = toGridLerp(point)
                paras = vec13(0.0)
                for k in ti.static(range(8)):
                    paras += scene[x+k % 2, y+k//2 % 2, z+k//4 % 2]*vol[k]
                tempColor, weight = SH_forward(paras, -ray, weight)
                color += tempColor
                dis += stride_length
            buffer[i, j] = color


@ti.kernel
def renderBuffer_back(buffer: ti.template(), scene: ti.template(), origins: ti.template(), rays: ti.template(), index: int, real_image: ti.template(), grad: ti.template()):
    pos = origins[index]
    for i, j in buffer:
        ray = rays[index, i, j]
        color_final = buffer[i, j]
        color_diff = color_final-real_image[i, j].xyz*real_image[i, j].w
        tmin, tmax = enterBox(pos, ray)
        dis = 0.0
        point = vec3(0.0)
        weight = 1.0
        color = vec3(0.0)
        ref_grad = vec13(color_diff[0], color_diff[1], color_diff[2],
                         -color_diff[0]*ray[0], -color_diff[1]*ray[0], -color_diff[2]*ray[0],
                         -color_diff[0]*ray[1], -color_diff[1]*ray[1], -color_diff[2]*ray[1],
                         -color_diff[0]*ray[2], -color_diff[1]*ray[2], -color_diff[2]*ray[2], 0.)
        if (tmin >= tmax) or (tmax < 0.0):
            buffer[i, j] = vec3(0.)
        else:
            if tmin < 0:
                tmin = 0.
            dis = tmin+0.1
            while (dis < tmax):
                point = pos+dis*ray
                x, y, z, vol = toGridLerp(point)
                paras = vec13(0.0)
                for k in ti.static(range(8)):
                    paras += scene[x+k % 2, y+k//2 % 2, z+k//4 % 2]*vol[k]
                tempColor, weight_new = SH_forward(paras, -ray, weight)
                color += tempColor
                color_left = color_final-color
                temp_grad = ref_grad*weight*(paras[12])
                temp_grad[12] = -(color_left@color_diff)*(weight*tm.pow(1.00001-paras[12], stride_length - 1)*paras[12])
                if real_image[i, j].w == 0.:
                    temp_grad = vec13(10000.)
                for k in ti.static(range(8)):
                    grad[x+k % 2, y+k//2 % 2, z+k//4 % 2] += temp_grad*vol[k]
                dis += stride_length
                weight = weight_new


@ti.kernel
def cluster(buffer: ti.template(), origins: ti.template(), rays: ti.template(), index: int, real_image: ti.template(), grad: ti.template()):
    pos = origins[index]
    for i, j in buffer:
        ray = rays[index, i, j]
        tmin, tmax = enterBox(pos, ray)
        dis = 0.0
        point = vec3(0.0)
        if (tmin >= tmax) or (tmax < 0.0):
            buffer[i, j] = vec3(0.)
        else:
            if tmin < 0:
                tmin = 0.
            dis = tmin+0.1
            while (dis < tmax):
                point = pos+dis*ray
                x, y, z, vol = toGridLerp(point)
                temp_grad = vec13(0.0)
                if real_image[i, j].w == 0.:
                    temp_grad = vec13(10000.)
                    for k in ti.static(range(8)):
                        grad[x+k % 2, y+k//2 % 2, z+k//4 % 2] = temp_grad*vol[k]
                dis += stride_length


if __name__ == '__main__':
    vs = voxel_system()
    camera = Camera(vs)
    vs.random_init()

    while camera.gui.running:
        camera.cal_rays()
        camera.render()
        camera.display()
        camera.processEvents()
