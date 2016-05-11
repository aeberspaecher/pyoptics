#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import pyoptics
from pyoptics.wpm import wpm_propagate



def maxwell_lens_func(x0, y0, z0, R, n_inside, n_outside):
    def maxwell(X, Y, z):
         n_vals = np.zeros_like(X)
         inside_inds = ((X-x0)**2 + (Y-y0)**2 + (z-z0)**2 <= R**2)
         #qimport pudb; pudb.set_trace()
         print("Number of inside indices: {}".format(np.sum(inside_inds)))
         n_vals[inside_inds] = n_inside
         n_vals[~inside_inds] = n_outside
         return n_vals

    return maxwell



Nx = 128
Ny = 128
Nz = 200

x_max = 500
y_max = 500
z_max = 200

maxwell = maxwell_lens_func(x0=0.0, y0=0.0, z0=100, R=50, n_inside=1.5, n_outside=1.0)


x = np.linspace(-x_max, +x_max, Nx)
y = np.linspace(-y_max, +y_max, Ny)
z = np.linspace(0, +z_max, Nz)

X, Y = np.meshgrid(x, y)

n = maxwell(X, Y, 100)

gauss_width = 50
wl = 0.630
field_plus = np.exp(-(X**2 + Y**2)/(22*gauss_width**2))

dz = z[1] - z[0]
for i in range(Nz):
    field_plus = wpm_propagate(maxwell, field_plus, x, y, z[i], dz, wl)
    plt.imshow(np.abs(field_plus)**2); plt.show()


#plt.imshow(n[:, Ny/2, :]); plt.colorbar(); plt.show()



#plt.imshow(maxwell(X, Y, 100)); plt.colorbar(); plt.show()
