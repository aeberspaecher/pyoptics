# WIP

## Low-hanging fruit

* rectangle with rounded egdes aperture

* rotation angle in apertures

* astigmatic beams
* Fresnel numbers for general annular apertures

## Medium

* Finish the stub `interpolator.TrigInterpolator`
* Lithography (DUV, see the roadmap in the project files): RCWA for 2D masks, ray traced stress
  birefringence
* More tests for `TIE`
* WPM: Fresnel transmission correction at index steps, absorbing boundaries;
  symmetrized step (alternate the two operator orderings) against the power
  drift in narrow waveguides (notebook 05, section 11)
* Docstrings: Parameters/Returns for single-argument functions, methods and
  properties (see the report-only numpydoc check in the docs workflow); then
  make the check blocking per module
* Documentation: user guide pages per topic, bibliography, doctests, weekly
  re-execution of the notebooks
* Incoherence in thin films (spatial and temporal incoherence)
* Basis class factory for numbering scheme plus callable

## More challenging

* Ince-Gaussian beams
* spectrum of plane wave propagation with analytical handling of linear phase
  and offset (cmp. Wyrowski)

## DONE

* Docstrings: NumPy-style Parameters/Returns sections for every public
  function with two or more arguments; report-only `numpydoc lint` in the docs
  workflow (configured in `pyproject.toml`)

* Rays to waves at the focus of a lithography lens (notebook 21): traced
  exit pupil plus angular spectrum through focus, ray caustics vs. wave
  caustics, axial colour; the angular spectrum propagators drop evanescent
  waves for z < 0

* Coated laser mirror (notebook 14): quarter wave layers at oblique incidence
  (`coatings.Layer.quarter_wave(angle=...)`), vectorial focus of polarization
  traced rays (`raytrace.System.vectorial_focus`)

* Radius of curvature measurement with a Fizeau interferometer (notebook 24):
  double pass systems (`raytrace.double_pass`), optical path on a detector
  grid (`System.ray_map`), displaced test spheres
  (`interferometry.fit_sphere_displacement`)

* Polarization ray trace with energy in `raytrace` (`System.trace_polarization`):
  coatings with thickness profiles on any surface (`coatings`), coated
  mirrors, absorbing, uniaxial (retardance, dichroism) and cubic crystal
  media (`materials`); `litho.lens.lens_jones_pupil` built on it

* Lithography: lens aberrations from the ray tracer across the scanner
  slit, Zernike sensitivities (`litho.lens`, notebook 33)
* Lithography: Hopkins TCC and SOCS imaging (`litho.hopkins`, notebook 34)
* Lithography: swing curves, BARC optimization and a resist model
  (exposure, PEB, Mack development; `litho.wafer`, `litho.resist`,
  notebook 35)
* Lithography: rigorous mask diffraction with 1D RCWA and boundary layer
  calibration (`litho.rcwa`, notebook 36)
* Lithography: Jones pupils (Pauli-Zernike, orientation Zernikes, polar
  decomposition) and the polarization ray trace of a lens with coatings
  and CaF2 intrinsic birefringence (`litho.jones_pupil`, `litho.lens`,
  notebook 38)
* Lithography: source optimization of pixelated sources (`litho.source_opt`,
  notebook 39)
* Lithography: OPC of Manhattan polygons (bias tables, SRAFs, model-based
  OPC with fragments and EPE; `litho.opc`, notebook 40)
* Lithography: geometric (chief ray) and energetic telecentricity of a
  coated lens across the slit, the real pupil and its apodization in the
  projector, pattern shift through focus (`litho.lens.lens_telecentricity`,
  `lens_apodization`, notebook 41)

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
* notebooks 17, 07 and 13 (PSF/MTF/Strehl, beam quality, stack transfer
  functions)
* vectorial propagation: E_z from div E = 0, vectorial angular spectrum,
  vectorized scalar propagators, Mansuripur's high NA focusing checked
  against Richards-Wolf (notebook 02)
* vectorial imaging (Abbe, s/p/unpolarized, lithography notebook 31),
  polarization converters x/y <-> TE/TM, Stokes -> Jones, DOP, IPS
* lithography toolbox `litho` (sources, mask layouts, thin and thin film
  mask stacks, boundary layer, resist on a thin film stack with TE/TM
  Fresnel transmission, NILS, CD, MEEF, Bossung curves, process windows;
  notebooks 31 and 32), pupil apodization (apodization Zernikes) and KrF
  material data
* ABCD matrices (`abcd`): rays, cardinal points, Gaussian beams via q,
  resonators, Collins integral (notebook 08)
* 2D phase unwrapping (`unwrap`): least squares (DCT and weighted PCG),
  quality guided, residues (notebook 22); reference guided (TIE) and
  two-wavelength unwrapping (notebook 27)
* interferometry (`interferometry`): phase-shifting algorithms (N-step,
  4-step, Carré, Hariharan), Takeda's Fourier-transform method,
  interferogram simulation, shifter miscalibration sensitivity (notebook 23)
* digital holography (`holography`): off-axis, phase-shifting and Gabor
  reconstruction, refocusing, autofocus, quantitative phase (notebook 26)
* chirp-z transform (`fft.czt()`, `zoom_FT()`, `zoom_inv_FT()`), shifted
  angular spectrum, scaled Fresnel/Fraunhofer propagators, far field in
  chosen directions, zoomed PSFs (notebook 04)
* sequential ray tracing (`raytrace`): spheres, conics, aspheres, mirrors,
  paraxial pupils and Seidel sums checked against `abcd`, OPD maps, Zernike
  fits, PSF/MTF/Strehl via `imaging`, spot diagrams and ray fans
  (notebook 19)
* tilts and decentres in `raytrace` (decentred/tilted surfaces, coordinate
  breaks, fold mirrors, tilted image planes), telecentric objects, the
  Caldwell KrF lithography lens as an example (notebook 19)
* axial (z) shift sensitivity of the Caldwell lens elements
  (`System.with_axial_shift`; focus, Z9, magnification; notebook 19)
* water immersion ArF lens (NA 1.35, 8x, 11 aspheres) from Zeiss patent
  US 7,187,503 B2, second embodiment (`raytrace.zeiss_arfi_lens()`); ray
  aiming by continuation for the rim rays of high NA lenses
* phase retrieval (`phase_retrieval`): error reduction and hybrid
  input-output with support, reality and positivity constraints,
  shrink-wrap, near-field (Fresnel) CDI via the propagators, phase
  diversity (multi-plane Gerchberg-Saxton and Zernike fit, paraxial or
  exact defocus) from through-focus PSFs; first tests for `ifta`
  (notebook 28)
* mode analysis and fibres (`modes`): HG/LG decomposition with fitted basis,
  M² from the mode content, LP modes of step-index fibres, Marcuse mode
  field radius, coupling efficiency and misalignment sweeps (notebook 09)
* propagation onto tilted planes (`tilted`): exact non-uniform sum and
  Matsushima's spectrum rotation, scalar and vectorial, tilted source to
  parallel target, beam evaluation (ISO 11146 widths) on inclined planes
  (notebook 03)
* Sphinx documentation (`docs/`, Furo theme, notebooks via myst-nb, API
  reference, built on every PR with warnings as errors, published to GitHub
  Pages)
