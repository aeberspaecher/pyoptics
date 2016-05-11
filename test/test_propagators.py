#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test diffraction patterns obtained from different propagators.
"""

import numpy as np
import matplotlib.pyplot as plt


import pyoptics as po
from pyoptics.utils import grid1d
from pyoptics.plot_tools import plot_image
from pyoptics.propagators import (fresnel_number, z_from_fresnel_number,
                                  fresnel_IR, fresnel_TF,
                                  rayleigh_sommerfeld_I_IR,
                                  rayleigh_sommerfeld_I_TF,
                                  rayleigh_sommerfeld_I_DI)


N = 1023
wl = 0.630  # e.g. micrometers
x_max = 9000
x_aperture_size = 200

x = grid1d((-x_max, +x_max), N, assume_periodicity=True)
y = grid1d((-x_max, +x_max), N, assume_periodicity=True)
X, Y = np.meshgrid(x, y)

aperture = np.zeros((N, N))
aperture[(np.abs(X) < x_aperture_size) & (np.abs(Y) < x_aperture_size)] = 1.0  # squre aperture
#aperture[(np.abs(x_aperture_size/4)**2 + np.abs(Y)**2 < x_aperture_size**2/8)] = 0.5

phase = (+0.0075*X + 0.00075*Y**2 + 0.00000003*X*Y)/wl
field = aperture*np.exp(1j*phase)  # add phase

#plot_image(aperture, x/wl, y/wl, xlabel="$x/\lambda$", ylabel="$y/\lambda$", title="Aperture")


phase[aperture == 0.0] = np.nan
#plot_image(phase, x/wl, y/wl, xlabel="$x/\lambda$", ylabel="$y/\lambda$", title="Phase")


F = 0.1  # Fresnel number to propagate to
z = z_from_fresnel_number(F, x_aperture_size, wl)
print("Propagating to z = {}".format(z))
print("Computing Fresnel TF diffraction")
E_Fresnel_TF, x_new, y_new = fresnel_TF(field, x, y, z, wl)
print("Computing Fresnel IR diffraction")
E_Fresnel_IR, x_new, y_new = fresnel_IR(field, x, y, z, wl)
print("Computing RS TF diffraction")
E_RS_TF, x_new, y_new = rayleigh_sommerfeld_I_TF(field, x, y, z, wl)
print("Computing RS IR diffraction")
E_RS_IR, x_new, y_new = rayleigh_sommerfeld_I_IR(field, x, y, z, wl)
print("Computing RS DI diffraction")
E_RS_DI, x_new, y_new = rayleigh_sommerfeld_I_DI(field, x, y, z, wl)

I_Fresnel_TF = np.abs(E_Fresnel_TF)**2
I_Fresnel_IR = np.abs(E_Fresnel_IR)**2
I_RS_TF = np.abs(E_RS_TF)**2
I_RS_IR = np.abs(E_RS_IR)**2
I_RS_DI = np.abs(E_RS_DI)**2

#I_Fresnel_TF /= np.max(I_Fresnel_TF)
#I_Fresnel_IR /= np.max(I_Fresnel_IR)
#I_RS_TF /= np.max(I_RS_TF)
#I_RS_IR /= np.max(I_RS_IR)
#I_RS_DI /= np.max(I_RS_DI)

from pyoptics.fft import save_wisdom
save_wisdom()

plot_image(I_Fresnel_TF, x/wl, y/wl, xlabel="$x/\lambda$",
           ylabel="$y/\lambda$", title="Fresnel TF", colorbar=True)
plot_image(I_Fresnel_IR, x/wl, y/wl, xlabel="$x/\lambda$",
           ylabel="$y/\lambda$", title="Fresnel IR", colorbar=True)
#plot_image(I_Fresnel_TF - I_Fresnel_IR , x/wl, y/wl, xlabel="$x/\lambda$",
           #ylabel="$y/\lambda$", title="Fresnel IR - TF", colorbar=True,
           #cmap=plt.cm.bwr)
plot_image(I_RS_TF, x/wl, y/wl, xlabel="$x/\lambda$",
           ylabel="$y/\lambda$", title="RS TF", colorbar=True)
plot_image(I_RS_IR, x/wl, y/wl, xlabel="$x/\lambda$",
           ylabel="$y/\lambda$", title="RS IR", colorbar=True)
#plot_image(I_RS_TF - I_RS_IR , x/wl, y/wl, xlabel="$x/\lambda$",
           #ylabel="$y/\lambda$", title="RS IR - TF", colorbar=True,
           #cmap=plt.cm.bwr)
plot_image(I_RS_DI, x/wl, y/wl, xlabel="$x/\lambda$",
           ylabel="$y/\lambda$", title="RS DI", colorbar=True)
plot_image(I_RS_DI - I_RS_IR , x/wl, y/wl, xlabel="$x/\lambda$",
           ylabel="$y/\lambda$", title="RS DI - IR", colorbar=True,
           cmap=plt.cm.bwr)
