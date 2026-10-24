# WIP

## Low-hanging fruit

* rectangle with rounded egdes aperture

* rotation angle in apertures

* astigmatic beams
* beam evaluation on inclined planes
* Fresnel numbers for general annular apertures

## Medium

* Finish the stubs: `interpolator.TrigInterpolator`,
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
* image processors: Scaler, CameraNoiser (EMVA 1288 camera model), Centerer
* padding/unpadding routines (`utils.pad()`, `unpad()`)
* image shift (`processing.Shifter`)
* rotation angle in basis functions, abstract base class for tensor product
  bases (`bases.TensorProductBasis`)
* Gaussian beam parameters according to ISO 11146 (`beam_quality`)
* analytical Airy disk and OTF/MTF for circular apertures
* PSF, OTF & MTF tool with pupil masks and phases, Strehl ratio and Maréchal
  approximations (`imaging`)
* transfer functions of thin film systems for the angular spectrum
  (`thin_films.stack_transfer_functions()`)
* notebooks 12 to 14 (PSF/MTF/Strehl, beam quality, stack transfer
  functions)
