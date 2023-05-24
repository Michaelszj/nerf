import taichi as ti
import taichi.math as tm

vec3 = ti.types.vector(3, ti.f32)
vec8 = ti.types.vector(8, ti.f32)
vec13 = ti.types.vector(13, ti.f32)
mat3 = ti.types.matrix(3, 3, ti.f32)
mat4 = ti.types.matrix(4, 4, ti.f32)
stride_length = 0.5
grid_l = 240
half_l = grid_l//2
l = grid_l/2-0.6


@ti.func
def sum(v):
    return v[0]+v[1]+v[2]


@ti.func
def SH_forward(p, d, w):
    color = vec3(0.0)
    color[0] = p[0]+p[3]*d[0]+p[6]*d[1]+p[9]*d[2]
    color[1] = p[1]+p[4]*d[0]+p[7]*d[1]+p[10]*d[2]
    color[2] = p[2]+p[5]*d[0]+p[8]*d[1]+p[11]*d[2]
    color *= w
    w *= tm.pow(1-p[12], stride_length)

    return color, w


@ti.func
def limit(x: ti.f32):
    return x*10000.0


@ti.func
def enterBox(pos, ray):
    t1 = 0.0
    t2 = 0.0
    t3 = 0.0
    t4 = 0.0
    t5 = 0.0
    t6 = 0.0
    if ray[0] != 0.0:
        t1 = (l-pos[0])/ray[0]
        t2 = (-l-pos[0])/ray[0]
    else:
        t1 = limit(l-pos[0])
        t2 = limit(-l-pos[0])
    if ray[1] != 0.0:
        t3 = (l-pos[1])/ray[1]
        t4 = (-l-pos[1])/ray[1]
    else:
        t3 = limit(l-pos[1])
        t4 = limit(-l-pos[1])
    if ray[2] != 0.0:
        t5 = (l-pos[2])/ray[2]
        t6 = (-l-pos[2])/ray[2]
    else:
        t5 = limit(l-pos[2])
        t6 = limit(-l-pos[2])

    x_min = tm.min(t1, t2)
    x_max = tm.max(t1, t2)
    y_min = tm.min(t3, t4)
    y_max = tm.max(t3, t4)
    z_min = tm.min(t5, t6)
    z_max = tm.max(t5, t6)

    t_min = tm.max(x_min, y_min, z_min)
    t_max = tm.min(x_max, y_max, z_max)

    return t_min, t_max


@ti.func
def toGrid(v):
    vt = v-vec3(0.5)
    x = ti.cast(tm.round(vt[0]), ti.i32)+half_l
    y = ti.cast(tm.round(vt[1]), ti.i32)+half_l
    z = ti.cast(tm.round(vt[2]), ti.i32)+half_l
    return x, y, z


@ti.func
def toGridLerp(v):
    vt = v-vec3(0.5)
    x = ti.cast(tm.floor(vt[0]), ti.i32)+half_l
    y = ti.cast(tm.floor(vt[1]), ti.i32)+half_l
    z = ti.cast(tm.floor(vt[2]), ti.i32)+half_l
    vt = vt-tm.floor(vt)
    t0 = 1-vt[0]
    t1 = 1-vt[1]
    t2 = 1-vt[2]
    return x, y, z, vec8(t0*t1*t2, vt[0]*t1*t2, t0*vt[1]*t2, vt[0]*vt[1]*t2, t0*t1*vt[2], vt[0]*t1*vt[2], t0*vt[1]*vt[2], vt[0]*vt[1]*vt[2])
