#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import skimage.data as skd
from skimage.color import rgb2gray
from scipy.signal import fftconvolve, hamming
import matplotlib.pyplot as plt

import skimage.restoration as r

import pyoptics as po

x = np.linspace(-1, +1, 512)
y = np.linspace(-1, +1, 512)
NA = 0.4
wavelength = 0.05
k = 2*np.pi/wavelength

hamming_window_1d = hamming(512)
hamming_window_2d = np.outer(hamming_window_1d, hamming_window_1d)

lena = sk.img_as_float(rgb2gray(skd.lena()))
lena /= np.max(lena)
lena *= hamming_window_2d

camera = sk.img_as_float(skd.camera())
camera /= np.max(camera)
#camera *= hamming_window_2d

#plt.imshow(camera, cmap= plt.cm.gray)
#plt.title("Object")
#plt.colorbar()
#plt.show()

obj = lena*np.exp(1j*(0.1*np.pi*(camera - 0.5)))
img = po.imaging.image(obj, x, y, NA=NA, k=k)
psf = po.imaging.get_psf(x, y, NA, k=k)

psf /= np.sqrt(np.sum(np.abs(psf)**2))
print("summed I_PSF = %s"%np.sum(np.abs(psf)**2))

#psf /= np.max(np.abs(psf))

print("Computing incoherent image")
img_incoh_conv = fftconvolve(np.abs(obj)**2, np.abs(psf)**2, mode="same")
print("Computing coherent image")
img_coh_conv = fftconvolve(obj, psf, mode="same")

# normalize images:
#img /= np.max(np.abs(img))
#img_incoh_conv /= np.max(np.abs(img_incoh_conv))

print("Plotting")

plt.ion()

plt.figure()
plt.imshow(np.abs(obj)**2, cmap=plt.cm.gray)
plt.colorbar()
plt.title("Object Intensity")
plt.show()

plt.figure()
plt.imshow(np.abs(psf)**2, cmap=plt.cm.gray)
plt.colorbar()
plt.title("PSF Intensity")
plt.show()

#plt.figure()
#plt.imshow(np.abs(img)**2, cmap=plt.cm.gray)
#plt.colorbar()
#plt.title("Intensity (from imaging)")
#plt.show()

#plt.figure()
#plt.imshow(np.angle(img), cmap=plt.cm.gray)
#plt.colorbar()
#plt.title("Phase (from imaging)")
#plt.show()

plt.figure()
plt.imshow(img_incoh_conv, cmap=plt.cm.gray)
plt.colorbar()
plt.title("Intensity incoherent (from convolution)")
plt.show()

#plt.figure()
#plt.imshow(np.abs(img_coh_conv)**2, cmap=plt.cm.gray)
#plt.colorbar()
#plt.title("Intensity coherent (from convolution)")
#plt.show()

#plt.figure()
#plt.imshow(np.angle(img_coh_conv), cmap=plt.cm.gray)
#plt.colorbar()
#plt.title("Phase (from convolution)")
#plt.show()

#plt.figure()
#plt.imshow(np.abs(img)**2 - np.abs(img_coh_conv)**2, cmap=plt.cm.seismic)
#plt.colorbar()
#plt.title("I_imgaging - I_convo")
#plt.show()

# try deconvolution:
#img_incoh_conv /= np.sum(img_incoh_conv)
#img_incoh_restored, _ = r.unsupervised_wiener(img_incoh_conv, np.abs(psf)**2,
                                              #clip=False)
img_incoh_restored = r.wiener(img_incoh_conv, np.abs(psf)**2, balance=2)
#img_incoh_restored, _ = r.richardson_lucy(img_incoh_conv, np.abs(psf)**2,
                                          #iterations=1)

plt.ioff()

plt.figure()
plt.imshow(img_incoh_restored, cmap=plt.cm.gray)
plt.colorbar()
plt.title("Deconvolution with incoherent image")
plt.show()

#plt.figure()
#plt.imshow(np.angle(img) - np.angle(img_coh_conv), cmap=plt.cm.seismic)
#plt.colorbar()
#plt.title("phi_imgaging - phi_convo")
#plt.show()

plt.close('all')
