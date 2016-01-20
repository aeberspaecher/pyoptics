Ideas, design
=============

General design
--------------

- object oriented where sensible

- Classes: Propagator, Imager, Source

- Utils: TIE, helpers related to CZT zoom factors, interpolator for complex
  fields (TODO: find FOSS code to incorporate!), stack_rescale

- BPM

- Vector wave propagation method (?)

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
  - classes for each ingredient (aperture mask, wavefront, apodisation) (allows
    to store a state [e.g. Zernike basis sizes, norms or ordering conventions] as
    well as easy usage [when the object are callable])
  - describe each defining component by only one vector (for optimization
    purposes - however, we might be able to write a calling function that
    does unpacking magic)

  Capabilities:
  - allow object and image side defocus (really?)
  - use correct radiometric correction
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
  - In a later (!) version, allow for vector aberrations (Jones and Müller
    matrices)

  Analysis:
  - PSF
  - OTF
  - MTF


Illumination pupil:
-------------------

- Define a source for imaging purposes by the source spectrum/illumination
  pupil. The pupil is defined in sigma space / parametrized by sigma.
- Information to use:
  - pupil mask geometry (circular/annular/dipole/quadrupole/disar/quasar)
  - polarization (TE/TM or x/y, degree of polarization)
  - intensity ([0;1])
- Create a routine that samples the pupil evenly
  (define number of samples per axis, let illumination pupil objects decide
  whether the sigma values are valid or not)

Object
------

Relevant for imaging is the field distribution in the front focal plane. Thus,
an object for imaging can be modelled by a function that returns the complex
valued field in the focal plane.

Wether the field got there from a self-radiating object or from a transparent
object being lit by an external source doesn'matter in the first place. The
imaging systems job is to define something that can deliever a field.

- Objects may be defined by transmittivity and phase shift ("Kirchhoff
  approximation"/"thin object approximation"/"ideal mask" approximation")


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

This class can be used in the imager for the pupil mask and specifically for
the wavefront and the apodisation.

Utils
-----

- TIE:
  - define all operators needed in their Fourier version
  - implement smoothing derivatives (because Scipy is fucking awesome it comes
    with a Savitzky-Golay filter!)
  - implement two TIEs: an ordinary one and the repeated one from recent L.
    Waller papers
- scalar product for 2d fields
- Waves and beams: Plane waves (with tilt, 1d and 2d), Gaussian beams, Airy
  beams...
- polarization converter (?): convert from Ex Ey representations to E_TM and
  E_TM ones - can this be done spectrally (ExEy -> TE,TM: is this more than a
  projection onto radial and tangential basis vectors in k-space)?
- reflection and transmission coefficients (for both complex fields and
  intensities) in both polarizations
- symmetrization and desymmetrization of images
- trigonometric interpolation of bandlimited images (enter NA, get interpolator
  that works on physical coordinates!)
- map: entries of parameter vector to values spatially resolved (e.g. for
  optmization in an SLM)
- Gerchberg-Saxton algorithm

Propagator
----------

- propagates fields
- carefully checks sampling!
- allows to choose how evanescent waves are dealt with
- implement different diffraction integrals in their Fourier formulation:
  - Rayleigh-Sommerfeld I
  - Fresnel
  - Fraunhofer
- where appropriate, both the transfer function propagator as well as the
  impulse response ones are implemented
- vectorial Fraunhofer?
- Smyte's formula?

Thin films
----------

Offer routines for reflectivity and transmittivity (both complex amplitude and
intensity) for thin film stacks.

Processors
----------

Processors are objects that take an image or an image stack process it. A prime
example would be the Scaler class that may scale images to a certain maximum
value, to a given energy or related.

- ideas for processors:
    - scaler (scale to one of max, energy, average)
    - shifter (subpixel by using FFT)
    - background remover (subtract average computed outside a given mask/ROI)
    - low-pass filter (defined by NA, performed by CZT)
    - noise adder (using a simple camera model)

Datasets
--------

Implement a data set - a compound data type that holds both data and
meta-information such as
- dimensionality
- the grid it is defined on (callable?)
- units
- flags (is_periodic, is_vector_valued, is_defined_on_uniform_grid, is_complex...)
- names

The awesome way would allow to use elementary operations such as +, -, *, : or
even complex ones (exp(), ...). Could we therefore sub-class np.ndarray?

Measurement functions
---------------------

Implement a few measurement tools centrally:

- max, avg and energy (with automagic switching to abs values if given complex
  fields?)
- cutlines (define positions, make routine interpolate the image along a line
  from starting position to end position): as measured images in optical
  systems are always bandlimited, it should be safe to use trigonometric
  interpolation (compute Fourier spectrum and store it, then use a truncated
  Fourier series for interpolation. Offer using symmetrized images?)

Dependencies
------------

- required: numpy, scipy, skimage
- optional: tfftw

Prerequisites and order of work
-------------------------------

Write those in roughly that order:

1. czt
2. Propagator
3. Basis, specifically Zernikes and a grating (define by two
   frequencies/lattice constants and two amplitudes)
4. Source and Imager


Fill utils as needed.

Side notes
----------

tfftw should probably have an pyfftw accelerated fftconvolve.

Ideas
-----

- rotations from czt as in that paper
- different phase tools (parabolic phase front, spherical ones)
- accuracy improved FFT as in Numerical Recipes
- Rayleigh-Sommerfeld direct integration as in that paper
- PSF tool as an iteractive IPython Notebook thingy
- simple aperture masks: circle, rectangle

- simmulated annealing with pixelated photomasks to arrive at target intensity distributions?


License
-------

Choose a license:

- BSD? BSD is awesome, but it is hard to protect code against the employer
  [not assured changes made on the job go back to the project]
- GPL? Better with the employer, but probably hard for everyone else.
- GPL with the option the re-license???
- LGPL?
