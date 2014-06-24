#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Module that defines image processing classes.

The following processors are defined:
- Scaler: scale images to given peak value, peak intensity or energy
- CamerNoiser: apply a simple noise model
- Shifter: shift images
- Centerer: center images
"""

class ImageProcessor(object):

    @abstractmethod
    def process_single_image(self, image):
        """Define how a single image is processed.
        """

        pass

    def __call__(self, data):
        # go through all images and process each image individually
        result = np.zeros_like(data)

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

    # TODO: make shift vectors properties
    # TODO: formulate in k-space

    pass


class Centerer(Shifter):
    """Processor that centers images.
    """

    # TODO: implement center of mass routines

    pass
