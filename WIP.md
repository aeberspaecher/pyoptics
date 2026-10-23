# WIP

## Low-hanging fruit

* rectangle with rounded egdes aperture

* padding/unpadding routines

* image shift

* rotation angle in apertures
* rotation angle in basis functions

* astigmatic beams
* abstract base class for tensor product bases
* beam evaluation on inclined planes
* Gaussian beam parameters extraction according to ISO
* Fresnel numbers for general annular apertures
* Analytical results:
  - OTF & MTF for circular apertures
  - Airy disk
* PSF & MTF tool (`imaging.get_psf()`: pupil masks and phases)
* Marechal & Strehl tool
* transfer function generator for thin film systems

## Medium

* Finish the stubs: `processing` (Scaler, CameraNoiser, Centerer),
  `interpolator.TrigInterpolator`, `utils.add_camera_noise()`,
  `polarization._polarization`
* Tests for `ifta` and more tests for `wpm` and `TIE`
* Documentation for propagators
* Incoherence in thin films (spatial and temporal incoherence)
* CZT
* Basis class factory for numbering scheme plus callable
* Polarization converters (x,y <-> TE, TM)
* Vectorial imaging (cmp. Jabbour et al)
* ABCD matrices

## More challenging

* Sphinx doc
* Ince-Gaussian beams
* spectrum of plane wave propagation with analytical handling of linear phase
  and offset (cmp. Wyrowski)

## DONE

* FFT module
* fitting bases to scattered data
* Chebyshev basis
* Müller matrices and some Stokes vectors
* elliptical aperture
* catch z=0 case in beam function
* "ensure_meshgrid()" function or similar
* modern packaging (pyproject.toml, replaces waf)
* Python 3 port
* test suite against analytical results, CI on Python 3.9, 3.11 and 3.13
* propagator tests against Mahajan's formula
* consistent phase convention exp(i(k·r − ωt)), documented in the README
* basic Jones vectors and matrices
* example notebooks
