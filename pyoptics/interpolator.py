#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Image interpolation.
"""

from fft import rFT


class ImageInterpolator(object):
    def __init__(self, image, x, y):
        self.image = image
        self.full_spectrum = rFT(image)

    def __call__(self, x, y):
        pass
# TODO: make base class abstract?


class TrigInterpolator(object):
    """Trigonometric interpolation for images recorded through an imaging
    system.

    A typical cutoff frequency would be e.g. 2*NA/wavelength for a fully
    incoherent imaging system.
    """

    def __init__ (self, image, x, y, f_cutoff):
        """ Class initialiser """

        super(TrigInterpolator, self).__init__(image, x, y)
        self.f_cutoff = f_cutoff

    @property
    def f_cutoff(self):
        return self.f_cutoff

    @f_cutoff.setter
    def f_cutoff(self, value):
        self.f_cutoff = f_cutoff
        self._recompute_spectrum()

    def _recompute_spectrum(self):
        F_X, F_Y = freq_grid(self.x, self.y, wavenumbers=False,
                             normal_order=True)
        allowed_freq_indices = (F_X <= self.f_cutoff) & (F_Y <= self.f_cutoff)
        self.F_X = F_X[allowed_freq_indices]
        self.F_Y = F_Y[allowed_freq_indices]
        self.truncated_image = self.image[allowed_freq_indices]

        # zeroify contributions of frequencies outside circular NA?
        if self.assume_circular_NA:
            self.truncated_image[F_X**2 + F_Y**2 > self.f_cutoff**2] = 0.0

    def __call__(self, x, y):
        pass

# TODO: how to do that?
# Option 1:
# truncate frequency spectra for x, y (also get_rid of unneeded elements)
# zerofiy spectrum outside the "NA circle" (but keep rectangular shape) [make
# that optional?]

# TODO: implement super-Gaussian filter?
