# WIP

## Low-hanging fruit

* rectangle with rounded egdes aperture

* rotation angle in apertures

* astigmatic beams
* beam evaluation on inclined planes
* Fresnel numbers for general annular apertures

## Medium

* Finish the stub `interpolator.TrigInterpolator`
* Lithography: rigorous mask diffraction (RCWA) to calibrate the boundary
  layer model, Hopkins/TCC (SOCS) imaging for speed, resist models beyond
  threshold and blur (development, PEB), OPC helpers
* Tests for `ifta` and more tests for `wpm` and `TIE`
* Documentation for propagators
* Incoherence in thin films (spatial and temporal incoherence)
* Basis class factory for numbering scheme plus callable

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
* vectorial propagation: E_z from div E = 0, vectorial angular spectrum,
  vectorized scalar propagators, Mansuripur's high NA focusing checked
  against Richards-Wolf (notebook 15)
* vectorial imaging (Abbe, s/p/unpolarized, lithography notebook 16),
  polarization converters x/y <-> TE/TM, Stokes -> Jones, DOP, IPS
* lithography toolbox `litho` (sources, mask layouts, thin and thin film
  mask stacks, boundary layer, resist on a thin film stack with TE/TM
  Fresnel transmission, NILS, CD, MEEF, Bossung curves, process windows;
  notebooks 16 and 17), pupil apodization (apodization Zernikes) and KrF
  material data
* ABCD matrices (`abcd`): rays, cardinal points, Gaussian beams via q,
  resonators, Collins integral (notebook 18)
* 2D phase unwrapping (`unwrap`): least squares (DCT and weighted PCG),
  quality guided, residues (notebook 21)
* interferometry (`interferometry`): phase-shifting algorithms (N-step,
  4-step, Carré, Hariharan), Takeda's Fourier-transform method,
  interferogram simulation, shifter miscalibration sensitivity (notebook 22)
* digital holography (`holography`): off-axis, phase-shifting and Gabor
  reconstruction, refocusing, autofocus, quantitative phase (notebook 23)
* chirp-z transform (`fft.czt()`, `zoom_FT()`, `zoom_inv_FT()`), shifted
  angular spectrum, scaled Fresnel/Fraunhofer propagators, far field in
  chosen directions, zoomed PSFs (notebook 20)
