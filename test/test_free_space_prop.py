#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


import pyoptics as po


x = np.linspace(-3, +3, 512)
y = np.linspace(-3, +3, 512)
X, Y = np.meshgrid(x, y)

lambd = 0.1
k = 2*np.pi/lambd

phase_ampl = 0.2  # fraction of 2pi

z = lambd

a = np.sqrt(np.exp(-(X/1.0)**2 - (Y/1.0)**2))
b = np.cos(1*X)*np.cos(2*Y)

obj = a*np.exp(1j*2*np.pi*phase_ampl*b)

obj_prop = po.propagators.prop_free_space(obj, x, y, z, k)

plt.imshow(np.abs(obj)**2, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Object intensity at z=0")
plt.show()

plt.imshow(np.abs(obj_prop)**2, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Object intensity at z=%s"%z)
plt.show()
