Ideas, design
=============

General design
--------------

- object oriented where sensible

- Classes: Propagator, Imager, Source

- Utils: TIE, helpers related to CZT zoom factors, interpolator for complex
  fields (TODO: find FOSS code to incorporate!), stack_rescale

Imager
------

- implements an idealized 4f system (defined by a source, an aperture
  and a complex pupil function)

  TODO: is it convenient to differentiate the mask defining the aperture,
  and two real valued functions for each apodisation and wavefront?
  --> probably yes as in that case different subclasses may take care of
      defining features separately (e.g. a subclass for typical optical systems
      with circular aperture stops)

  Parameters defining the basic system:
  - NA defining the pupil dimension
  - number of pixels
  - pixel size in each object and image space (related by a magnifaction factor)
  - wavelengths

  Use:
  - Chirp z transform as a zoom FFT for better pupil sampling
  - classes for each ingredient (mask, wavefront, apodisation) (allows to store
    a state [e.g. Zernike basis sizes, norms or ordering conventions] as well as
    easy usage [when the object are callable])
  - describe each defining component by only one vector (for optimization
    purposes - however, we might be able to write a calling function that
    does unpacking magic)

  Capabilities:
  - allow object and image side defocus
  - compute a PSF
  - compute images
    - how to do that in a sensible way? Incoherent: from PSF? Coherent:
      convolution with the PSF or by going to the pupil using a CZT?
      Arguments for PSF:  may be needed anyway (incoherent imaging)
      Against PSF: for large NA, we may seriously lose information when the PSF
      extends over only a few pixels [can we help ourselves with a CZT here?
      Or do we just want to sample a bit better?]
  - perspectively is able to deal with polarization/vectorial imaging as well
  - checks sampling and prints a warning in case the grid is such that
    pixel size > Airy diameter
  - partial coherence: implement Abbe's method, later Hopkins etc.

Illumination pupil:
-------------------

- allow definition of an illumination pupil as known from lithography tools
  (in sigma space - with polarization per point, Jones matrices per point...)
- How to use this in imaging? Generate k_x, k_y from an appropriate sampling
  recipe?

Source
------

- Define a source for imaging purposes by the source spectrum (defines pupil
  filling!)
- Which information do we need?
  - Spatial extent
  - Phase?? Some other information related to coherence (sigma?)?
  - Intensity?
- Does it make sense to define a source by electric field strength at the object
  plane?

Object
------

- Objects may be defined by transmittivity and phase shift.

Other abstractions
------------------

A very useful abstraction is that of a Basis class - a set of basis functions
that may be evaluated, i.e. a map of a parameter vector to values defined on a
rectangular grid.

Methods:

- __init__(): takes arguments following arguments and sets state
  - basis_size
  - N_x, N_y
- is callable with a parameter vector, returns field evaluated on grid
- fit_to_field(): take a field and perform a least-squares fit of basis to
  field (alternatively evaluate scalar product c_i = <i|field>

This class can be used in the imager for the mask and specifically for the
wavefront and the apodisation.

Utils
-----

- TIE:
  - define all operators needed in their Fourier version
  - implement smoothing derivatives (because Scipy is fucking awesome it comes
    with a Savitzky-Golay filter!)
- scalar product for 2d fields
- Waves and beams: Plane waves (with tilt, 1d and 2d), Gaussian beams, Airy
  beams...

Propagator
----------

- propagates fields
- carefully checks sampling!

Processors
----------

Processors are objects that take an image or an image stack process it. A prime
example would be the Scaler class that may scale images to a certain maximum
value, to a given energy or related.

Dependencies
------------

- required: numpy, scipy, skimage
- optional: tfftw

Prerequisites and order of work
-------------------------------

Write those in roughly that order:

1. czt
2. Basis, specifically Zernikes and a Grating (define by two
   frequencies/lattice constants and two amplitudes)
3. Source and Imager
5. Propagator

Fill utils as needed.

Side notes
----------

tfftw should probably have an pyfftw accelerated fftconvolve.
