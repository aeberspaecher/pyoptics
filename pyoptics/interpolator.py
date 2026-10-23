#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Image interpolation.
"""

import numpy as np

from .fft import FT
from .utils import freq_grid


class ImageInterpolator(object):
    def __init__(self, image, x, y):
        self.image = image
        self.x = x
        self.y = y
        self.full_spectrum = FT(image)

    def __call__(self, x, y):
        pass
# TODO: make base class abstract?


class TrigInterpolator(ImageInterpolator):
    """Trigonometric interpolation for images recorded through an imaging
    system.

    A typical cutoff frequency would be e.g. 2*NA/wavelength for a fully
    incoherent imaging system.
    """

    def __init__ (self, image, x, y, f_cutoff, assume_circular_NA=True):
        """ Class initialiser """

        super(TrigInterpolator, self).__init__(image, x, y)
        self.assume_circular_NA = assume_circular_NA
        self.f_cutoff = f_cutoff

    @property
    def f_cutoff(self):
        return self._f_cutoff

    @f_cutoff.setter
    def f_cutoff(self, value):
        self._f_cutoff = value
        self._recompute_spectrum()

    def _recompute_spectrum(self):
        F_X, F_Y = freq_grid(self.x, self.y, wavenumbers=False,
                             normal_order=True)
        spectrum = self.full_spectrum.copy()

        # zeroify contributions of frequencies outside circular NA?
        if self.assume_circular_NA:
            spectrum[F_X**2 + F_Y**2 > self.f_cutoff**2] = 0.0

        # keep the rectangular region of frequencies below the cutoff:
        allowed_freq_indices = (np.abs(F_X) <= self.f_cutoff) & (np.abs(F_Y) <= self.f_cutoff)
        self.F_X = F_X[allowed_freq_indices]
        self.F_Y = F_Y[allowed_freq_indices]
        self.truncated_spectrum = spectrum[allowed_freq_indices]

    def __call__(self, x, y):
        pass

# TODO: how to do that?
# Option 1:
# truncate frequency spectra for x, y (also get_rid of unneeded elements)
# zerofiy spectrum outside the "NA circle" (but keep rectangular shape) [make
# that optional?]

# TODO: implement super-Gaussian filter?
