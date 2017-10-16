#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import pyoptics
from pyoptics.wpm import wpm_propagate
from pyoptics.propagators import *
from pyoptics.utils import simpson_weights, weight_grid



def spherical_lens_func(x0, y0, z0, R, n_inside, n_outside):
    def spherical(X, Y, z):
         n_vals = np.zeros_like(X)
         inside_inds = ((X-x0)**2 + (Y-y0)**2 + (z-z0)**2 <= R**2)
         n_vals[inside_inds] = n_inside
         n_vals[~inside_inds] = n_outside
         return n_vals
         #return np.ones(np.shape(X))

    return spherical



Nx = 71
Ny = 71
Nz = 700

x_max = 120
y_max = 120
z_max = 100

n_lens = 1.5

spherical = spherical_lens_func(x0=0.0, y0=0.0, z0=70, R=70, n_inside=n_lens, n_outside=1.0)


x = np.linspace(-x_max, +x_max, Nx, endpoint=True)
y = np.linspace(-y_max, +y_max, Ny, endpoint=True)
z = np.linspace(0, +z_max, Nz)

X, Y = np.meshgrid(x, y)

gauss_width = 30
wl = 0.630
field_plus = np.exp(-(X**2 + Y**2)/(2*gauss_width**2))

print("dz / wl", (z[1] - z[0])/wl)

print("Suggested x sampling: Nx = %s"%suggest_lateral_sampling(2*x_max, n_lens, wl))
print("Suggested y sampling: Ny = %s"%suggest_lateral_sampling(2*y_max, n_lens, wl))



## compare to other diffraction routine:
#field_propagated, _, _ = rayleigh_sommerfeld_I_IR(field_plus, x, y, z[100], wl)
#plt.imshow(np.abs(field_propagated)**2); plt.colorbar(); plt.show()


dz = z[1] - z[0]
for i in range(Nz):
    field_plus = wpm_propagate(spherical, field_plus, x, y, z[i], dz, wl)
    n_curr = spherical(X, Y, z[i])
    plt.imshow(n_curr, cmap=plt.cm.gray, interpolation="none"); plt.colorbar(); plt.savefig("n_%i.png"%i); plt.clf()
    #plt.imshow(n_curr, cmap=plt.cm.gray, alpha=0.7)
    plt.imshow(np.abs(field_plus)**2, interpolation="none"); plt.colorbar(); plt.savefig("%i.png"%i); plt.clf()#plt.show()
    print(i)


#plt.imshow(n[:, Ny/2, :]); plt.colorbar(); plt.show()



#plt.imshow(spherical(X, Y, 100)); plt.colorbar(); plt.show()
