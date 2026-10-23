#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Module that defines image processing classes.

The following processors are defined:
- Scaler: scale images to given peak value, peak intensity or energy
- CamerNoiser: apply a simple noise model
- Shifter: shift images
- Centerer: center images
"""

# TODO: how to deal with processors that need to extract parameters for
# processing a single image for each image in stack individually? how can we use
# * magic in a sensible way? certainly storing a state does not make any sense
# here.

# that's actually not a problem! in that case, we can just make the
# process_single_image() routine do that.

from abc import abstractmethod

import numpy as np

from .fft import fft2, ifft2
from .utils import freq_grid


class ImageProcessor(object):
    """Base class for all image processor.

    All derived classes need to
    - have a constructor that correctly sets the processor up
    - implement process_single_image(self, image, *params)
    """

    @abstractmethod
    def process_single_image(self, image, *params):
        """Define how a single image is processed.

        The image may either be a complex valued field or an intensity.

        Parameters
        ----------
        image : array

        """

        pass

    def __call__(self, data, *params):
        """Process a single image (2d array) or an image stack (3d array, the
        last index numbering the images).
        """

        if np.ndim(data) == 2:
            return self.process_single_image(data, *params)

        # go through all images and process each image individually
        result = np.zeros_like(data)
        for i in range(np.size(data, 2)):
            result[:, :, i] = self.process_single_image(data[:, :, i], *params)

        # TODO: distinhuish tuples/lists from arrays?

        return result


class Scaler(ImageProcessor):
    """Processor that scales an image or an image stack.
    """

    pass


class CameraNoiser(ImageProcessor):
    """Processor that applies a very simple noise nodel to an image stack.
    """

    pass


class Shifter(ImageProcessor):
    """Processor that shifts images.
    """

    def __init__(self, delta_x, delta_y, x, y):
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.x = x
        self.y = y

    def process_single_image(self, img):
        Img = fft2(img)
        # fft2() returns the spectrum in standard order:
        K_X, K_Y = freq_grid(self.x, self.y, wavenumbers=True, normal_order=False)

        # use the Fourier transform's shift property (periodic shift):
        img_shifted = ifft2(np.exp(-1j*(K_X*self.delta_x + K_Y*self.delta_y))*Img)

        if(np.iscomplexobj(img)):
            return img_shifted
        else:
            return np.real(img_shifted)


class Centerer(Shifter):
    """Processor that centers images.
    """

    # TODO: implement center of mass routines in utils module

    pass
