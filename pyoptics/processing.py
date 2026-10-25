#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Module that defines image processing classes.

The following processors are defined:

- Scaler: scale images to a given peak, mean or total value of the field
  strength or of the intensity
- CameraNoiser: apply a camera noise model (Gaussian, Poisson photon noise or
  a full linear camera model following EMVA 1288)
- Shifter: shift images
- Centerer: center images on their intensity centroid

All processors are callables that accept a single image (2d array) or an
image stack (3d array, the last index numbering the images). Images may be
complex valued fields E or (real) intensities. The intensity of a field is
taken as ``|E|**2`` throughout this module, without the prefactor n/(2 Z_0) of
utils.I().
"""

from abc import abstractmethod

import numpy as np

from .fft import fft2, ifft2
from .utils import center_of_mass, freq_grid


class ImageProcessor(object):
    """Base class for all image processor.

    All derived classes need to

    - have a constructor that correctly sets the processor up
    - implement ``process_single_image(self, image, *params)``
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

        # go through all images and process each image individually; the
        # result may have a different dtype than the input (e.g. noisy float
        # images from integer counts)
        return np.stack([self.process_single_image(data[:, :, i], *params)
                         for i in range(np.size(data, 2))], axis=2)


def _is_field(image, image_type):
    """Decide whether image is a field (True) or an intensity (False)."""

    if image_type == "auto":
        return np.iscomplexobj(image)
    if image_type == "field":
        return True
    if image_type == "intensity":
        return False
    raise ValueError("image_type must be 'auto', 'field' or 'intensity'")


def _intensity(image, image_type="auto"):
    """Intensity ``|E|**2`` of a field, or the image itself if it is an intensity."""

    if _is_field(image, image_type):
        return np.abs(image)**2
    return np.asarray(image, dtype=float)


class Scaler(ImageProcessor):
    """Processor that scales an image or an image stack.

    The image is multiplied by a real factor such that a chosen measure of
    either the field strength ``|E|`` or the intensity I = ``|E|**2`` equals a given
    value. Images may be fields or intensities: a field is multiplied by c, an
    intensity by c**2 if the field strength is to be scaled by c.

    Parameters
    ----------
    value : float
        Target value of the measure.
    measure : str, optional
        "peak" (maximum), "mean" or "total". For the intensity, "total" is
        the power sum(I)*pixel_area; for the field strength it is the norm
        ``sqrt(sum(|E|**2)*pixel_area)``, so that scaling the field to a total of
        sqrt(P) is the same as scaling the intensity to a total of P.
    quantity : str, optional
        "intensity" or "field": the quantity whose measure is set to value.
    image_type : str, optional
        What the input images are: "field", "intensity" or "auto" (complex
        images are fields, real images are intensities). Use "field" for real
        valued amplitudes.
    per_image : bool, optional
        If True (default), every image of a stack is scaled individually. If
        False, one common factor is used for the stack so that relative
        intensities are preserved; the measure is then taken over the whole
        stack (global peak, global mean, total averaged over the images).
    pixel_area : float, optional
        Pixel area dx*dy used for measure="total".

    Examples
    --------
    >>> to_photons = Scaler(1000.0)   # peak intensity: 1000 photons per pixel
    >>> unit_power = Scaler(1.0, measure="total", pixel_area=dx*dy)
    """

    measures = ("peak", "mean", "total")
    quantities = ("intensity", "field")

    def __init__(self, value, measure="peak", quantity="intensity",
                 image_type="auto", per_image=True, pixel_area=1.0):
        if measure not in self.measures:
            raise ValueError("measure must be one of {}".format(self.measures))
        if quantity not in self.quantities:
            raise ValueError("quantity must be one of {}".format(self.quantities))
        _is_field(np.zeros(1), image_type)  # validate image_type

        self.value = value
        self.measure = measure
        self.quantity = quantity
        self.image_type = image_type
        self.per_image = per_image
        self.pixel_area = pixel_area

    def get_measure(self, data):
        """Return the chosen measure of the chosen quantity of an image or (for
        a stack) of the whole stack as used with per_image=False.
        """

        intensity = _intensity(data, self.image_type)
        if self.quantity == "intensity":
            q = intensity
        else:
            q = np.sqrt(intensity)

        if self.measure == "peak":
            return np.max(q)
        if self.measure == "mean":
            return np.mean(q)

        # total: sum over the image, averaged over the images of a stack
        num_images = np.size(intensity, 2) if np.ndim(intensity) == 3 else 1
        power = np.sum(intensity)*self.pixel_area/num_images
        return power if self.quantity == "intensity" else np.sqrt(power)

    def get_factor(self, data):
        """Return the factor the image (field or intensity) is multiplied by."""

        m = self.get_measure(data)
        if m == 0:
            raise ValueError("cannot scale an image whose {} {} is zero".format(
                             self.measure, self.quantity))

        factor = self.value/m  # factor for the chosen quantity
        image_is_field = _is_field(data, self.image_type)
        if image_is_field and self.quantity == "intensity":
            factor = np.sqrt(factor)
        elif not image_is_field and self.quantity == "field":
            factor = factor**2

        return factor

    def process_single_image(self, image):
        return self.get_factor(image)*image

    def __call__(self, data):
        if np.ndim(data) == 3 and not self.per_image:
            return self.get_factor(data)*data

        return super().__call__(data)


class CameraNoiser(ImageProcessor):
    """Processor that applies a camera noise model to images.

    Three models are available:

    - "gaussian": additive white Gaussian noise of standard deviation sigma
      (absolute, or relative to the image's peak if relative=True). The image
      may be an intensity or a field; for a complex field, independent noise
      is added to the real and imaginary parts.
    - "poisson": photon shot noise only. The image is the expected number of
      photons per pixel; the result is a Poisson distributed photon count.
    - "camera": linear camera model following the EMVA 1288 standard. The
      image is the expected number of photons per pixel and exposure. The
      chain is ::

          photons --(Poisson, quantum_efficiency)--> photo electrons
          + dark electrons (Poisson, dark_current*exposure_time)
          -> clipped at full_well
          * (1 + PRNU pattern)
          + read noise (Gaussian, read_noise) + DSNU pattern
          * gain + offset
          -> rounded and clipped to [0, 2**bit_depth - 1] (if bit_depth)

      The result is in digital numbers (DN). The fixed pattern noise (PRNU,
      DSNU) is drawn once per image shape and then stays the same for all
      images processed by this noiser, like on a real sensor.

    For the "poisson" and "camera" models, fields are converted to intensities
    ``|E|**2`` first. Use a Scaler to bring images to photon numbers, e.g.
    Scaler(1000.0) for a peak of 1000 photons per pixel.

    Parameters
    ----------
    model : str, optional
        "gaussian", "poisson" or "camera".
    sigma : float, optional
        Standard deviation of the "gaussian" model.
    relative : bool, optional
        If True, sigma is relative to the peak of each image ("gaussian"
        model only).
    quantum_efficiency : float, optional
        Fraction of photons converted to electrons.
    dark_current : float, optional
        Dark current in electrons per pixel and unit time.
    exposure_time : float, optional
        Exposure time (in the time unit of dark_current).
    read_noise : float, optional
        Read noise (temporal dark noise) in electrons rms.
    full_well : float, optional
        Full well capacity in electrons; None means no saturation.
    gain : float, optional
        System gain in DN per electron (K in EMVA 1288).
    offset : float, optional
        Dark offset (bias) in DN.
    bit_depth : int, optional
        ADC resolution in bits; None means no quantization or clipping.
    prnu : float, optional
        Photo response non-uniformity: relative rms gain variation from pixel
        to pixel.
    dsnu : float, optional
        Dark signal non-uniformity: rms offset variation from pixel to pixel
        in electrons.
    rng : int, numpy.random.Generator or None, optional
        Seed or random number generator for reproducible noise.
    """

    models = ("gaussian", "poisson", "camera")

    def __init__(self, model="camera", sigma=0.0, relative=False,
                 quantum_efficiency=1.0, dark_current=0.0, exposure_time=1.0,
                 read_noise=0.0, full_well=None, gain=1.0, offset=0.0,
                 bit_depth=None, prnu=0.0, dsnu=0.0, rng=None):
        if model not in self.models:
            raise ValueError("model must be one of {}".format(self.models))

        self.model = model
        self.sigma = sigma
        self.relative = relative
        self.quantum_efficiency = quantum_efficiency
        self.dark_current = dark_current
        self.exposure_time = exposure_time
        self.read_noise = read_noise
        self.full_well = full_well
        self.gain = gain
        self.offset = offset
        self.bit_depth = bit_depth
        self.prnu = prnu
        self.dsnu = dsnu
        self.rng = np.random.default_rng(rng)
        self._patterns = {}  # image shape -> (prnu pattern, dsnu pattern)

    def fixed_pattern(self, shape):
        """Return the fixed pattern noise (relative gain deviation, offset
        deviation in electrons) for images of the given shape.
        """

        shape = tuple(shape)
        if shape not in self._patterns:
            prnu = self.prnu*self.rng.standard_normal(shape)
            dsnu = self.dsnu*self.rng.standard_normal(shape)
            self._patterns[shape] = (prnu, dsnu)

        return self._patterns[shape]

    def mean_signal(self, photons):
        """Expected output of the "camera" model in DN for the given expected
        photon numbers, ignoring fixed pattern noise, saturation and
        quantization.
        """

        electrons = self.quantum_efficiency*np.asarray(photons) \
                    + self.dark_current*self.exposure_time
        return self.gain*electrons + self.offset

    def temporal_noise(self, photons):
        """Expected temporal noise (standard deviation) of the "camera" model
        in DN for the given expected photon numbers (EMVA 1288 eq. for the
        linear model, without quantization noise).
        """

        electrons = self.quantum_efficiency*np.asarray(photons) \
                    + self.dark_current*self.exposure_time
        return self.gain*np.sqrt(electrons + self.read_noise**2)

    def process_single_image(self, image):
        if self.model == "gaussian":
            return self._gaussian(image)

        photons = _intensity(image)
        if np.any(photons < 0):
            raise ValueError("photon numbers must not be negative")

        if self.model == "poisson":
            return self.rng.poisson(photons).astype(float)

        return self._camera(photons)

    def _gaussian(self, image):
        sigma = self.sigma*np.max(np.abs(image)) if self.relative else self.sigma
        noise = sigma*self.rng.standard_normal(np.shape(image))
        if np.iscomplexobj(image):
            noise = noise + 1j*sigma*self.rng.standard_normal(np.shape(image))

        return image + noise

    def _camera(self, photons):
        electrons = self.rng.poisson(self.quantum_efficiency*photons) \
                    + self.rng.poisson(self.dark_current*self.exposure_time,
                                       size=photons.shape)
        electrons = electrons.astype(float)
        if self.full_well is not None:
            electrons = np.minimum(electrons, self.full_well)

        prnu, dsnu = self.fixed_pattern(photons.shape)
        electrons = electrons*(1 + prnu) + dsnu \
                    + self.read_noise*self.rng.standard_normal(photons.shape)

        signal = self.gain*electrons + self.offset
        if self.bit_depth is not None:
            signal = np.clip(np.round(signal), 0, 2**self.bit_depth - 1)

        return signal


class Shifter(ImageProcessor):
    """Processor that shifts images.

    The shift uses the Fourier shift theorem: it is periodic (content leaving
    the grid on one side enters it on the other side) and works for subpixel
    shifts. For real images the real part of the result is returned.
    """

    def __init__(self, delta_x, delta_y, x, y):
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.x = x
        self.y = y

    def shift(self, img, delta_x, delta_y):
        """Shift img by (delta_x, delta_y)."""

        Img = fft2(img)
        # fft2() returns the spectrum in standard order:
        K_X, K_Y = freq_grid(self.x, self.y, wavenumbers=True, normal_order=False)

        # use the Fourier transform's shift property (periodic shift):
        img_shifted = ifft2(np.exp(-1j*(K_X*delta_x + K_Y*delta_y))*Img)

        if(np.iscomplexobj(img)):
            return img_shifted
        else:
            return np.real(img_shifted)

    def process_single_image(self, img):
        return self.shift(img, self.delta_x, self.delta_y)


class Centerer(Shifter):
    """Processor that centers images on their intensity centroid.

    Every image is shifted (with the periodic Fourier shift of Shifter) such
    that its intensity centroid (center of mass, see utils.center_of_mass())
    lies at (x0, y0). Fields are centered on the centroid of ``|E|**2``.

    Parameters
    ----------
    x, y : arrays
        1d coordinate grids of the images (image shape (len(y), len(x))).
    x0, y0 : float, optional
        Target position of the centroid; default is the origin.
    threshold : float, optional
        Pixels with an intensity below threshold times the image's peak are
        ignored when computing the centroid. A threshold suppresses the
        influence of background and noise far away from the spot.
    per_image : bool, optional
        If True (default), every image of a stack is centered individually,
        e.g. to remove beam jitter. If False, the whole stack is shifted by one
        common shift that centers the centroid of the summed intensity, which
        preserves the relative positions of the images (e.g. a drifting focus).
    iterations : int, optional
        Number of centroid/shift passes. Since the shift is periodic, the
        first pass may be inexact for spots near the edge or with strong
        background; further passes refine the result.
    """

    def __init__(self, x, y, x0=0.0, y0=0.0, threshold=0.0, per_image=True,
                 iterations=1):
        super().__init__(0.0, 0.0, x, y)
        self.x0 = x0
        self.y0 = y0
        self.threshold = threshold
        self.per_image = per_image
        self.iterations = iterations

    def centroid(self, image):
        """Return the intensity centroid (x_c, y_c) of an image."""

        return center_of_mass(_intensity(image), self.x, self.y,
                              threshold=self.threshold)

    def get_shift(self, image):
        """Return the shift (delta_x, delta_y) that centers an image."""

        x_c, y_c = self.centroid(image)
        return self.x0 - x_c, self.y0 - y_c

    def get_shifts(self, data):
        """Return the shifts (one row (delta_x, delta_y) per image) applied to
        the images of a stack in the first pass.
        """

        if np.ndim(data) == 2:
            return np.array([self.get_shift(data)])
        if not self.per_image:
            summed = np.sum(_intensity(data), axis=2)
            return np.tile(self.get_shift(summed), (np.size(data, 2), 1))

        return np.array([self.get_shift(data[:, :, i])
                         for i in range(np.size(data, 2))])

    def process_single_image(self, img):
        for _ in range(self.iterations):
            img = self.shift(img, *self.get_shift(img))

        return img

    def __call__(self, data):
        if np.ndim(data) == 3 and not self.per_image:
            for _ in range(self.iterations):
                delta_x, delta_y = self.get_shift(np.sum(_intensity(data), axis=2))
                data = np.stack([self.shift(data[:, :, i], delta_x, delta_y)
                                 for i in range(np.size(data, 2))], axis=2)
            return data

        return super().__call__(data)
