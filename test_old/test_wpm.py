#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from pyoptics.wpm import wpm_propagate, averaging_z_sampling
from pyoptics.propagators import suggest_lateral_sampling


def spherical_lens_func(x0, y0, z0, R, n_inside, n_outside):
    def spherical(X, Y, z):
        n_vals = np.zeros_like(X)
        inside_inds = ((X-x0)**2 + (Y-y0)**2 + (z-z0)**2 <= R**2)
        n_vals[inside_inds] = n_inside
        n_vals[~inside_inds] = n_outside
        return n_vals

    return spherical


Nx = 201
Ny = 201
Nz = 500

x_max = 200
y_max = 200
z_max = 200

n_lens = 1.5


x = np.linspace(-x_max, +x_max, Nx, endpoint=True)
y = np.linspace(-y_max, +y_max, Ny, endpoint=True)
z = np.linspace(0, +z_max, Nz)
dz = z[1] - z[0]

X, Y = np.meshgrid(x, y)

lens_func = spherical_lens_func(x0=0.0, y0=0.0, z0=70, R=70, n_inside=n_lens,
                                n_outside=1.0)  # lens_func is callable with
                                                # (x, y, z) as arguments

# averaged_lens_func should also be callable with (x, y, z) as arguments
averaged_lens_func = lambda X, Y, z: averaging_z_sampling(lens_func,
                                                          X, Y, z, z+dz, 5)


gauss_width = 30
wl = 0.630
field_plus = np.exp(-(X**2 + Y**2)/(2*gauss_width**2))

print("dz / wl", (z[1] - z[0])/wl)

print("Suggested x sampling: Nx = %s"%suggest_lateral_sampling(2*x_max, n_lens,
                                                               wl))
print("Suggested y sampling: Ny = %s"%suggest_lateral_sampling(2*y_max, n_lens,
                                                               wl))

for i in range(Nz):
    field_plus = wpm_propagate(averaged_lens_func, field_plus, x, y, z[i], dz,
                               wl)
    n_curr = averaged_lens_func(X, Y, z[i])
    plt.imshow(n_curr, cmap=plt.cm.gray, interpolation="none")
    plt.colorbar()
    plt.savefig("n_%i.png"%i)
    plt.clf()
    #plt.imshow(n_curr, cmap=plt.cm.gray, alpha=0.7)
    I = np.abs(field_plus)**2
    plt.imshow(I, interpolation="none")
    plt.title("Energy = {:.2f}".format(np.sum(I)))
    plt.colorbar()
    plt.savefig("%i.png"%i)
    plt.clf()
    print(i)

# TODO: implement testcase from literature (Maxwell fisheye, Luneberg lens or prism with Gaussian)
