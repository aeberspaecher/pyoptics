# Modules

What each module does and how mature it is. Planned work is listed in
[`WIP.md`](https://github.com/aeberspaecher/pyoptics/blob/master/WIP.md).

- `propagators`: mature and tested (Rayleigh-Sommerfeld, Fresnel, Fraunhofer,
  band-limited and padded propagation; vectorial propagation of [E_x, E_y]
  with the longitudinal component E_z, any scalar propagator vectorized,
  Mansuripur's high NA focusing, checked against the Richards-Wolf integrals;
  propagation onto free output windows: shifted angular spectrum method with
  Matsushima's band limit, scaled Fresnel and Fraunhofer integrals, far field
  in chosen directions)
- `tilted`: propagation onto tilted planes, tested: scalar and vectorial
  fields from z = 0 onto a plane with any rotated normal and back (tilted
  source to parallel target), sampled in the plane's own coordinates; exact
  non-uniform plane wave sum (separable for tilts about x or y, finufft or a
  direct sum otherwise) and Matsushima's spectrum rotation with the
  Jacobian; carrier removal, power flux and ISO 11146 beam widths on
  inclined planes (detectors); Wigner and Husimi functions of 1D fields
  (`wigner()`, `husimi()`, the latter checked as the smoothed Wigner
  function); `tilted_plane()` and
  `vectorial_tilted_plane()` are also available from `propagators`
- `fdfd`: new, tested: finite-difference frequency-domain solver in 2D
  (Ez and Hz) on the Yee grid of `fdtd` with its shapes and media (and a
  graded background permittivity given as a function, e.g. a GRIN lens;
  checked against the exact field of Maxwell's fish-eye), a
  stretched-coordinate PML and Bloch boundaries; driven solves (line
  currents, plane waves injected on a total-field box without leakage),
  resonances (quasi-normal modes and Q by shift-invert Arnoldi) and mode
  tracing through a change of the structure by field overlap; far fields
  from outgoing cylindrical waves and the directionality of the emission;
  checked against the cylinder series and the exact microdisk modes
  (`analytical.disk_resonance`), the slab under oblique incidence and the
  FDTD (notebooks 66 (FDFD basics) and 67 (limaçon microcavity)); Husimi functions of Ez modes on the cavity
  boundary in Birkhoff coordinates (`husimi_birkhoff()`, notebook 104, Husimi functions and ray chaos)
- `fdtd`: new, tested: finite-difference time-domain
  solver for Maxwell's equations in 1D, 2D and 3D (2D polarizations Ez and Hz,
  with the TE/TM naming explained in the module docstring), NumPy kernels
  and compiled numba kernels (optional, about ten times faster, identical
  results), shapes with sub-cell permittivity averaging (anisotropic for
  the Hz mode), conductive and dispersive media (Drude, Lorentz, Debye,
  fits to n + ik data),
  convolutional PML, PEC and periodic boundaries, point and line current
  sources and a total-field/scattered-field plane wave, probes, snapshots
  (float16, ring buffer, streaming to HDF5), running DFTs and flux
  monitors, plots and animations (also of the steady state from one DFT);
  tilted line sources for oblique beams; checked against `thin_films`
  (Bragg mirror, lossy, Drude, Lorentz and Debye films) and the exact
  cylinder series in both polarizations (also for a Drude metal);
  waveguides: a mode solver that is exact on the Yee grid, one-way mode
  sources (TFSF), mode monitors with forward/backward amplitudes and
  S-parameters, waveguide shapes (paths, S-bends, bends, rings) and
  resonance analysis by the matrix pencil method (Q factors), checked
  against the analytic slab waveguide, supermodes of coupled waveguides and
  the Fabry-Perot modes of a slab; periodic structures: Bloch boundaries
  (oblique incidence at a fixed transverse wave number), diffraction
  orders of gratings and band structures of photonic crystals
  (`bloch_modes`), checked against `thin_films`, RCWA (`litho.rcwa`), the
  analytic bands of a Bragg stack and the band gap of a rod lattice;
  devices: lattices with defects (`Lattice`), photonic-crystal
  waveguides (supercell bands, transmission through a sharp bend) and
  cavities (Q against the crystal size), a fibre-to-chip grating coupler,
  complex transmission of unit cells (`DiffractionOrders.coefficient`)
  for a metasurface phase library and a metalens; 3D: all six components
  (float32 and numba by default, CPML stored only in the layers), 3D
  shapes (`Sphere`, `Box`, `Cylinder`, `Prism` extrusions of 2D shapes),
  plane sources, a TFSF plane wave along z with the incident spectrum
  recorded, plane monitors, flux through planes and section plots,
  checked against the vector Mie series (near field and scattering cross
  section of a sphere); the photonic nanojet; diffraction orders (m, n) of 3D periodic cells;
  `Medium.from_index` for metals at one wavelength (an exact Drude match);
  the rigorous FDTD mask model `litho.FDTDMask`
- `analytic_phase`: propagation with analytic handling of the wavefront
  phase after Wyrowski and coworkers, tested: fields stored as a smooth
  sampled amplitude times a separable quadratic phase with carrier
  (`QuadraticPhaseField`); exact semi-analytical Fourier transforms (SFT)
  with automatic switch to the FFT; angular spectrum and Fresnel propagation
  with the same small number of samples at any distance, through the focus;
  propagation onto tilted planes (spectrum rotation with only the smooth
  amplitude interpolated); pointwise homeomorphic Fourier transform and
  geometric propagation with amplitude transport and Gouy phase
- `fieldtrace`: field tracing after Wyrowski and Kuhn (2011), new: a
  `Field` of E_x, E_y as smooth amplitude times analytic phase (scalar with
  a Jones vector while globally polarized, E_z, H and the Poynting vector
  on demand); free-space operators (spectrum of plane waves and Fresnel via
  the SFT, far field, geometrical optics) chosen automatically per step from
  1D cuts (error against the converged SPW result, effort = converged
  samples), components (thin lens, thin element, aperture, interface, Jones
  matrices, any callable), detectors and a `Chain` with a step-by-step
  report; polarization "auto" switches between scalar and vector fields;
  `Trace(system)` and `Chain.from_system()` take a field through a
  `raytrace.System` (polarization ray trace, ray field summation) and hand
  over to wave optics behind it; `Parallel` arms (interferometers) and
  `Obstruction` (Babinet parts on their own grid) carry a `FieldSum` whose
  parts are added at the detector; operator "fresnel_ft" (single-FT
  Fresnel) for small, strongly diffracting fields; `Field.overlap(mode)`
  and the detector quantity "coupling" give the power coupling efficiency
  into a mode (e.g. of a single-mode fibre)
- `polygons`: new, tested: polygons as (N, 2) vertex arrays: signed area,
  centroid and second moments, the exact Fourier transform of any polygon
  (Lee and Mittra's edge sum, with a Taylor series near k = 0 and vertices
  taken relative to their mean, so clips far from the origin keep their
  digits; shared by `segmented`, `photography` and `litho.mask`), even-odd
  point containment, Sutherland-Hodgman clipping to a window and wire
  outlines along polylines (`path_polygon()`, miter joins as GDSII paths)
- `fft`, `utils`, `masks`, `geometry`, `analytical`: mature and tested
  (`fft.czt()`: Bluestein's chirp-z transform; `fft.zoom_FT()` and
  `zoom_inv_FT()` evaluate `FT()` and `inv_FT()` on any equidistant grid)
  (`utils.pad()`/`unpad()` for centered zero padding; `analytical` holds the
  Airy pattern, its encircled energy, the diffraction limited OTF and the
  exact series solutions for a dielectric cylinder (TE) and sphere (scalar),
  also for beams: `cylinder_field()`, `sphere_field()`, and the vector
  Lorenz-Mie solution of a sphere: `mie_coefficients()`, `mie_field()`,
  `mie_efficiencies()`, and its far field `mie_amplitudes()` (S1, S2, checked
  against the optical theorem and the scattering efficiency))
- `bases`: mature (Zernike, Legendre, Chebyshev bases, numerical
  orthogonalization, fits); all bases can be shifted and rotated, Legendre and
  Chebyshev derive from the abstract `TensorProductBasis`; tested
- `beams`: tested (plane, spherical, Hermite-Gaussian and Laguerre-Gaussian
  beams)
- `interfaces`, `thin_films`: mature and tested (Fresnel coefficients,
  coherent multilayer stacks; `stack_transfer_functions()` gives r and t on
  the angular spectrum grid of a field)
- `materials`: Abbe (n_d, V_d) dispersion model via a two-term Cauchy
  formula, the index of water (Daimon and Masumura, 20 deg C) and of
  standard air (Ciddor), and of GeO₂-doped silica (Fleming's Sellmeier
  data for SiO₂ and GeO₂ with a linear mixing rule), tested
- `rainbow`: new, tested: rays through a spherical drop with any number of
  internal reflections (exact vector trace, Fresnel factors for s and p),
  Descartes' angles, the geometric cross section of every order, Airy's
  theory of the bow, and the drop as a sequential `raytrace.System` (one
  tilted surface per intersection); checked against the deviation
  formula, the envelope theorem, a Monte Carlo of traced rays, energy
  conservation and the Mie series (notebook 71, rainbows and glories)
- `schlieren`: new, tested: schlieren, shadowgraph and the knife-edge tests
  of Foucault and Ronchi as Fourier plane filters: knife edge, slit, wire
  and Ronchi ruling on the focal plane grid, coherent images of a point
  source and incoherent images of slit and pinhole sources, the geometric
  limit (transverse aberration through the filter), Toepler's contrast
  f eps/a and the linear knife-edge transfer for weak wavefronts (slope
  response for coarse structure, phase response for fine structure), knife
  defocus, a conic mirror at its centre of curvature and its zonal offsets,
  the Gladstone-Dale index of hot air and the shadowgraph -L laplace(W);
  checked against the Hilbert transform image of a knife at the focus, the
  knife-edge rule, Couder zone readings, Ronchi band periods and ripple
  contrast (notebook 76, Schlieren and knife-edge)
- `colour`: new, tested: CIE 1931 colour matching functions (analytic fit
  of Wyman, Sloan and Shirley), XYZ and sRGB of spectra, black body spectra
- `glasses`: optical glass catalogues of SCHOTT, OHARA, HOYA, CDGM, SUMITA
  and HIKARI (about 1600 glasses with the vendors' Sellmeier and power
  series formulas, from the refractiveindex.info database, CC0), tested:
  a `Glass` is a material n(wavelength_nm) for `raytrace.Surface`, Abbe
  numbers and relative partial dispersions for any lines, the normal line
  and dP_gF, catalogue search by n_d/V_d (and dP_gF), filters by status,
  index and Abbe number, secondary spectrum of thin achromats; real glasses
  for `lens_optimization.DampedLeastSquares.snap_glasses()` (notebook 61, glass catalogues)
- `polarization.mueller`, `polarization.jones`: Mueller calculus and basic
  Jones vectors and matrices, tested
- `vectorial_imaging`: vectorial, partially coherent imaging (Abbe's method)
  of 1D and 2D masks with conventional, annular, dipole and quasar sources,
  TE/TM/unpolarized/partially polarized illumination, contrast and NILS,
  tested (two-beam contrast of TE and TM waves)
- `litho`: lithography imaging toolbox built on `vectorial_imaging`,
  tested: illumination sources (conventional, annular, dipole, quadrupole,
  quasar, multipole with sector or circular poles, freeform maps) for any
  sigma and any polarization (x, y, TE/azimuthal, TM/radial, unpolarized,
  Jones, Stokes, degree of polarization); mask layouts with analytic
  diffraction orders (lines and spaces, isolated lines, trenches, dense,
  isolated and staggered contact holes, any rectangles, clear or dark
  field; polygons at any angle with exact Fourier coefficients from
  `polygons.polygon_ft()`, and `MaskLayout.from_gds()` for the cells,
  polygons and dicts of gdstk and gdspy with layer selection, a window as
  the period and a unit scale, without a GDSII reader of its own); mask stacks in the thin mask approximation (binary, attenuated
  PSM), with absorber transmissions from `thin_films` (angle and TE/TM
  dependent) and an optional boundary layer model for thick mask edges;
  projector with immersion (or dry), vectorial or scalar model, Fringe
  Zernike aberrations and pupil apodization (amplitude transmission
  1 + sum of Fringe Zernikes, optionally as log amplitude, or any
  transmission function); images in the resist on a thin film wafer stack (BARC, top
  coat, substrate; TE/TM Fresnel transmission, standing waves); metrics
  (contrast, ILS/NILS, threshold CD with linear or Fourier interpolation,
  dose to size, MEEF, focus-exposure
  matrix, Bossung curves, process window, exposure latitude vs. depth of
  focus, Gaussian resist blur) and plots (source, mask, pupil fill, images,
  Bossung curves, process windows); `litho.lens` hands a ray traced lens
  (`raytrace`) to the projector (Fringe Zernikes on the projector's pupil,
  distortion, telecentricity, placement of a small feature), computes CD,
  placement and best focus across a scanner slit with scan averaging and
  the linear scanner correction, and Zernike sensitivities with a quick
  aberration budget; `litho.hopkins` computes transmission cross
  coefficients (TCC) and their SOCS decomposition, equal to Abbe's method
  for thin masks and fast for many masks with the same optics; swing
  curves and BARC optimization (`litho.wafer`, absorbed energy exact
  against `thin_films`) and `litho.resist`, a chemically amplified resist
  in an x-z cross section (exposure, PEB with acid diffusion, quencher and
  amplification, Mack development with a fast marching front; profile, CD
  at top/middle/bottom, sidewall angle, top loss, dose to size, E0);
  `litho.rcwa`: rigorous mask diffraction of 1D layouts by RCWA (Li's
  factorization, conical incidence, enhanced transmittance matrices; exact
  against `thin_films`, energy conserving), `RigorousMask` for Abbe and
  Hopkins imaging and a calibration of the boundary layer model;
  `litho.FDTDMask`: rigorous mask diffraction of any 1D or 2D layout
  (contact holes, line ends) with the 3D FDTD, the same interface as
  `RigorousMask`, checked against RCWA for lines (orders, images, CD);
  `litho.jones_pupil`: polarization of the lens as Jones pupils in the
  representations of the literature (Pauli-Zernike, polar decomposition,
  retardance and diattenuation in orientation Zernike polynomials) with
  fits, used by `Projector(jones_pupil=...)` in Abbe and Hopkins imaging,
  and a polarization ray trace of coated or uncoated lenses with optional
  CaF2 elements and their intrinsic birefringence
  (`litho.lens.lens_jones_pupil` on top of `raytrace.System.trace_polarization`,
  `materials.CubicCrystal`);
  `litho.source_opt`: source optimization of a pixelated source with
  symmetry, pupil fill and common dose constraints, maximizing the worst
  NILS over clips and foci by linear programming on the Abbe responses;
  `litho.opc`: optical proximity correction of Manhattan polygons
  (`MaskLayout.from_polygons`, exact unions of rectangles): bias tables
  through pitch, SRAF rules with a printability check, fragmentation, edge
  placement errors and model-based OPC with a regularized Jacobian update
  (Abbe, Hopkins or SOCS imaging); `litho.illuminator`: a ray traced KrF
  illuminator for the Caldwell lens (zoom-axicon of US 5,675,401 adapted
  to fused silica, diffuser, glass rod, REMA objective of US 5,982,558):
  sigma 0.32 to 0.83 and annular settings, pupil fill per field point as
  an `Illumination`, field uniformity, telecentricity against the lens
- `polarization`: TE/TM bases, Stokes to Jones decomposition, DOP and IPS
- `imaging`: scalar imaging, tested: coherent imaging through a circular
  pupil, PSF with pupil masks (obscuration, apodization) and phases (also
  zoomed onto any output window via the chirp-z transform), OTF and
  MTF, Strehl ratio and Maréchal approximations
- `microscopy`: microscope imaging, tested: partially coherent bright
  field images of thin objects after Abbe, with Köhler illumination sampled
  on the frequency grid (disk, annulus, oblique or any source points,
  `source_points()`), pupils with exact defocus, aberrations and masks
  (`pupil()`), weak object amplitude and phase transfer functions
  (`weak_object_transfer()`) and the grating cutoff λ/(NA + NA_c); Zernike
  phase rings (`phase_ring()`) and the Nomarski DIC effective object with
  shear and bias (`dic_object()`); scalar 3D amplitude PSFs from the pupil
  (uniform or aplanatic, `psf_3d()`), fluorescence confocal PSFs for any
  pinhole (`confocal_psf()`, Airy units), the thin plane response of
  optical sectioning and the reflection mirror response; Fourier
  ptychography: LED array illumination, the low resolution forward model
  and the reconstruction by alternating projections with embedded pupil
  recovery (quasi-Newton or EPRY step), pupil overlap in k-space
- `tweezers`: new, tested: optical forces on a dielectric sphere in a
  focused beam (optical tweezers), as trapping efficiencies Q = Fc/(n P);
  Ashkin's closed form ray efficiencies (checked against an explicit trace
  of the internal reflections) and the ray optics force of a fan of rays
  from an aplanatic pupil with s/p splitting per ray; the Richards-Wolf
  focus as a plane wave quadrature (E, H, gradients, beam power); the exact
  force on a sphere of any size (`MieTrap`: every plane wave scattered by
  Mie theory, the force from the momentum lost in the far field, with an
  optional numba kernel); the dipole force with the radiatively corrected
  polarizability, the plane wave radiation pressure efficiency; equilibrium,
  stiffness, overdamped Brownian motion and the Lorentzian power spectrum;
  checked against the dipole force (deviation ∝ (ka)²), Mie radiation
  pressure, the Poynting flux of the focus and ray optics for large beads
  (notebook 75, optical tweezers)
- `superresolution`: new, tested: super-resolution fluorescence microscopy;
  SIM patterns and raw images, band separation from the pattern phases
  (`sim_bands()`), band shifts by any wave vector, the generalized Wiener
  reconstruction and the OTF support for several angles; vectorial vortex
  foci with any polarization (`vortex_focus()`, the true zero needs spin
  parallel to the orbital angular momentum), the effective STED PSF and the
  square root law; pixelated Gaussian spots, centroid, least squares and
  Poisson maximum likelihood localization, Mortensen's precision formulas
  (notebook 78, super-resolution microscopy)
- `abcd`: ABCD (ray transfer) matrices, tested: thin and thick lenses,
  spherical interfaces, tilted mirrors (tangential/sagittal), GRIN media;
  `OpticalSystem` builds systems element by element and traces rays and
  Gaussian beams (beam radius, wavefront radius, Gouy phase) anywhere, also
  inside elements; cardinal points and images; complex beam parameter q and
  the ABCD law (also in media), mode matching efficiency; resonator
  stability, eigenmodes and round trip Gouy phase; `collins()` propagates any
  field through an ABCD system (Collins integral, separate x and y matrices
  for astigmatic systems)
- `unwrap`: 2D phase unwrapping, tested: least squares unwrapping with a
  DCT Poisson solver (Ghiglia-Romero), weighted least squares with
  modulation or amplitude weights and masks (conjugate gradients
  preconditioned by the DCT solver), quality guided unwrapping by
  reliability sorting (Herráez), unwrapping with a reference estimate
  (e.g. a TIE phase) and two-wavelength (synthetic wavelength) unwrapping,
  residue maps, `wrap()` and a 1D helper
- `interferometry`: fringe analysis, tested: simulated interferograms
  (Zernike test wavefronts, tilt carrier, visibility, camera noise via
  `processing.CameraNoiser`, miscalibrated phase shifters); phase-shifting
  algorithms (N-step least squares for arbitrary known shifts, 4-step,
  Carré with unknown step, Hariharan's 5-step), each returning the wrapped
  phase and the modulation; Takeda's Fourier-transform method for single
  interferograms with a carrier (sideband window, automatic carrier search,
  masks, apodization, overlap warning); phase error vs. shifter
  miscalibration
- `coherence`: new, tested: spatial and temporal coherence and intensity
  interferometry; the van Cittert-Zernike theorem for any incoherent source
  (chirp-z transform of its intensity), the degree of coherence of uniform
  and linearly limb darkened stellar disks and of binaries, random
  realizations of incoherent sources; gamma(tau) of a sampled spectrum
  (Wiener-Khinchin), line shapes and coherence times of Gaussian,
  Lorentzian and rectangular lines; fringe visibility and phase by least
  squares; chaotic light with a given line shape, pairs with a given degree
  of coherence, g1, g2 and normalized intensity correlations (Hanbury Brown
  and Twiss); checked against 2J1(v)/v, the limb darkening formula, the
  Siegert relation and random-phase sources propagated with
  `fresnel_scaled()` (notebook 74, coherence and stellar interferometry)
- `oct`: new, tested: optical coherence tomography; spectral interferograms
  of a Michelson interferometer with DC, autocorrelation and cross terms, a
  spectrometer model (pixel averaging and Gaussian resolution), A-scans by
  FFT with the depth axis and z_max = pi/(2 delta k), time-domain signals
  for a scanned reference and their Hilbert demodulation, r_S(k) of layer
  stacks via `thin_films` (dispersive, absorbing) and the single
  reflection amplitudes, full range interferograms from a quarter wave
  reference step, round trip dispersion phases, Taylor coefficients,
  numerical compensation and its optimization by A-scan sharpness; checked
  against the axial resolution 2 ln2/pi lambda^2/Delta lambda, the
  sensitivity roll-off, the dispersion broadening of a Gaussian spectrum
  and the SD/TD SNR ratio (notebook 82, OCT)
- `holography`: digital holography, tested: off-axis and in-line holograms
  (tilted reference, phase shifts; camera noise via `processing.CameraNoiser`),
  sampling condition for the reference angle; reconstruction by sideband
  filtering (circular, Tukey, Hann, Gaussian windows, automatic carrier
  search), phase-shifting holography (via `interferometry.psi_n_step()`) and
  Gabor holography with iterative twin image suppression; zero order
  suppression; numerical refocusing (angular spectrum or Fresnel, padded);
  autofocus (amplitude variance, Tamura coefficient, gradient, Laplacian);
  quantitative phase with background compensation, tilt removal and
  unwrapping (`unwrap`)
- `raywave`: conversion between ray ensembles and wave fields, tested:
  ray field summation from a structured ray grid (ray tubes with the
  amplitude from the ray density, overlapping sheets added with their Gouy
  (Maslov) phases), Gaussian beamlets launched from rays or from sampled
  fields (also from an `analytic_phase.QuadraticPhaseField`) and propagated
  in closed form through foci and caustics, or traced through the surfaces
  of a `raytrace.System` (base ray and parabasal rays, complex ABCD law,
  Gouy phase on the continuous branch), rays along the local phase
  gradient (traceable with `raytrace.System.trace()`), rays from local
  spectra (sampled Husimi function) and the Wigner function of a 1D field;
  structured rays from fields for the round trip through ray-traced optics
- `billiard`: new, tested: ray dynamics in convex 2D billiards bounded by
  an `fdtd.PolarShape` (limaçon, `fdtd.Stadium`, circle); the bounce map
  in Birkhoff coordinates (area preserving and time reversible to 10⁻⁷ and
  10⁻¹⁰), Poincaré surfaces of section, the open dielectric billiard with
  Fresnel losses (steady probability distribution, escape directions and
  the ray far field) and boundary Husimi functions of a wave from its
  values and normal derivatives (Hentschel, Schomerus and Schubert),
  checked against analytic whispering gallery waves (notebook 104, Husimi functions and ray chaos)
- `grin`: new, tested: ray tracing through gradient-index media (the ray
  equation in the parameter dt = ds/n, Runge-Kutta-Nyström after Sharma,
  Kumar and Ghatak, with the optical path and exact end points on a stop
  surface); media with closed form rays: Luneburg lens, Maxwell's
  fish-eye, parabolic and hyperbolic secant rods, an exponential layer
  (mirage), or any index function with a numerical gradient; checked
  against all closed forms (notebook 77, GRIN optics)
- `cloak`: new, tested: carpet cloaks from quasi-conformal maps (Li and
  Pendry): the map above a bump on a mirror from two Laplace solves on a
  boundary-fitted grid with sliding, mirror-sliding or fixed boundaries,
  the isotropic index n² = det K and the dropped anisotropy, first-order
  theory of the index, the index as an FDFD permittivity and a `grin`
  medium with rays reflected on the bump and refracted at the cloak's
  edge; a line of FDFD sources that launches a Gaussian beam with the
  grid's dispersion, and angular spectra (notebook 80, carpet cloak)
- `illumination`: new, tested: illumination optics with Monte Carlo rays;
  Lambertian sources on disks and rectangles (`LambertianSource`, rays for
  `raytrace.System.trace()`, the Sun with its angular radius), canonical
  phase space coordinates (x, y, nL, nM), the skew invariant, the ray map of
  a `raytrace.System` and its Jacobian with a symplecticity check; étendue
  of Lambertian sources, the concentration limit n²/sin²α, the coupling
  bound and the stagnation temperature under concentrated sunlight; étendue
  from rays (occupied phase space volume and the participation étendue
  P²/∫L², unbiased against Monte Carlo noise); refraction at a plane, thin
  diffusers and straight light pipes traced by unfolding (TIR or mirror
  walls); checked against 2w·NA, 4π√det Σ, det J = 1 for a real lens and
  cos⁴θ′ for the paraxial ideal lens, and the brightness bound
  NA′²/sin²α of a stigmatic lens (notebook 83, étendue); compound parabolic
  concentrators (edge-ray design, hollow or dielectric with TIR walls, 2D
  trough or 3D, truncated) traced through their wall reflections, with a
  compiled kernel if numba is installed; checked against the ideal 2D step,
  the exit étendue under Lambertian input and the 1/n² bound for a
  dielectric CPC that exits into air (notebook 84, nonimaging concentrators); backward ray tracing
  through the reversed system (`reverse_system()`, `backward_trace()`,
  `backward_irradiance()`), checked against forward Monte Carlo and
  π sin²θ behind a stop (notebook 85, Köhler and critical illumination); ideal thin lenses and lenslet
  arrays for rays (`thin_lens()`, `lenslet_array()`) and the scalar wave
  field of a fly's eye homogenizer (`fly_eye_field_1d()`), checked against
  the flat top p F/f, its independence of the input tilt and the
  interference period λF/p (notebook 86, homogenizers); LED emitters with the pattern
  cos^m θ (also from quasi-random samples), irradiance maps and far-field
  intensity profiles from rays, the irradiance of a point source on a
  plane and of a Lambertian disk on its axis, and a paraboloidal
  reflector; checked against cos^m θ, E = I_0 cos^(m+3) θ/h², the
  photometric distance and the brightness bound I = L A of an LED
  collimator (notebook 88, LED optics)
- `speckle`: new, tested: speckle statistics and speckle reduction; rough
  surfaces with Gaussian heights and correlation, coherent images through
  a circular pupil, random diffuser illumination of a given NA; contrast,
  the gamma densities of summed patterns and the intensity autocorrelation
  1 + |2J1(v)/v|²; the contrast of correlated sums from the eigenvalues of
  the correlation matrix, polarization diversity √((1 + P²)/2), the
  wavelength decorrelation exp(−(Δk σ_L)²) and the contrast for a Gaussian
  or sampled spectrum, the limit of angle diversity from the mutual
  intensity of the illumination (0.678 NA_obs/NA_ill for large ratios) and
  C² = C∞² + (1 + C∞²)/M for M diffuser positions; checked against wave
  optics simulations (notebook 87, speckle reduction)
- `freeform`: new, tested: beam shaping by ray mapping; the radial ray map
  from equal enclosed power, Fermi–Dirac flat tops, the two-lens Gaussian
  to flat-top shaper (Kreuzer, Hoffnagle–Jefferson) from equal optical path
  and Snell's law, the optimal transport map between two irradiances on
  grids (log-domain Sinkhorn with a separable kernel), least squares
  integration of a gradient field, a plano-freeform lens that sends a
  collimated beam along a given map, and `TabulatedSurface`, a tabulated
  radial or x–y sag for the ray tracer; checked against the closed-form
  map, collimation and equal optical path, separable maps and traced
  irradiance (notebook 89, freeform beam shaping)
- `radiometry`: new, tested: radiometry and photometry; black bodies in
  absolute units (`planck_radiance()`, Stefan–Boltzmann, Wien), the
  spectral luminous efficiencies V(λ) and V'(λ) as analytic fits (the CIE
  tables are referenced, not shipped), luminous flux and the luminous
  efficacy of spectra and black bodies, photon fluxes, solid angles, view
  factors between Lambertian surfaces, the integrating sphere (the sphere
  multiplier and a Monte Carlo trace through the bounces), the camera
  equation with the working F-number, exposure values, reflected light
  metering and photons per pixel; checked against Stefan–Boltzmann, the
  CIE tables, the solar constant, reciprocity and Monte Carlo (notebooks
  105 (radiometry and photometry) and 106 (radiometry of images))
- `photography`: new, tested: camera lens optics, the out-of-focus image
  (bokeh); a 50 mm f/1.8 double Gauss with clear apertures that vignette
  (`double_gauss()`), focusing to the smallest spot (`focus()`), blur
  diameters from the thin lens formula and from the pupils (exact in
  first order), rays from a point through the whole aperture with the
  vignetting of all apertures and an optional apodizer, binned into images
  (`point_footprint()`, `splat()`, `relative_illumination()`), the
  paraxial vignetting circles and the cat's eye, the geometric irradiance
  of a blur with defocus and spherical aberration and the same from the
  FFT of the pupil, a rippled surface for the ray trace (onion rings) with
  the ring contrast from rays and from the Talbot effect, Gaussian
  apodizers and the T-number; checked against the paraxial trace, ray
  histograms, wave optics and the Airy pattern (notebook 90, bokeh); sunstars:
  iris openings with straight or rounded blades (`iris()`, area from the
  F-number), the exact Fourier transform of a polygon as a sum over its
  edges (`polygon_transform()`), the far field of the antialiased opening
  by FFT (`iris_psf()`), the spike laws (2n spikes for odd n, n for even
  n; (L/kA)² along a spike; the crossover 2πρ/L² of rounded blades and
  their fans; power outside k, P/(πAk), for any shape) and colour images
  of the Sun through an iris on a sensor (`sunstar()`); checked against
  the exact transform, the Airy pattern and the edge theory (notebook 91, sunstars);
  flare and ghosts: every double reflection path of a lens unfolded into
  a `raytrace.System` (`ghost_system()`) and traced with the polarization
  ray trace and the thin film coatings of its surfaces (`trace_flare()`,
  `ghost_footprint()` with adaptive sampling, a polygonal iris and the
  Sun's disk), uncoated, MgF2 and multilayer antireflection designs
  (`ar_coatings()`), the paraxial ghost analysis (`ghost_analysis()`) and
  spectral ghost images (`ghost_spectra()`); checked against the product
  of the Fresnel factors, the paraxial ghost irradiance and the full
  polarization trace (notebook 92, flare and ghosts); depth of field and tilted lenses:
  thin lens depth of field limits and the hyperfocal distance
  (`dof_limits()`), the wave optical MTF of defocus (`defocus_mtf()`) and
  the optimum F-number, a tilted thin lens with the hinge and Scheimpflug
  rules (`plane_of_focus()`, `tilted_blur()`), and a real lens tilted with
  coordinate breaks (`tilt_lens()`, `trace_point()`); checked against the
  traced double Gauss, the diffraction and geometric MTFs and Merklinger's
  depth of field wedge (notebook 94, depth of field and tilt-shift); a 28 mm f/2.8 retrofocus lens
  (`retrofocus()`), the real chief ray mapping (`chief_ray_mapping()`)
  and the relative illumination with the image side area factor
  (`relative_illumination(image_side=True)`), checked against cos⁴ for an
  ideal lens (notebook 106, radiometry of images)
- `sensor`: new, tested: image sensors; the pixel aperture MTF and the
  sampling of the image (`sensor_image()`: lens diffraction, an optical
  low pass filter and the pixel aperture on a fine grid), the slanted edge
  MTF of ISO 12233 (`slanted_edge_mtf()`), the Bayer mosaic with bilinear
  and Malvar–He–Cutler demosaicing and its chrominance carriers, the zone
  plate, the walk-off of the extraordinary ray in a uniaxial crystal
  (closed form and from the dielectric tensor), birefringent low pass
  filters of plates and wave plates as displaced Jones vectors
  (`olpf_spots()`, `olpf_otf()`), and a ray trace of a pixel with a
  microlens against the chief ray angle with microlens shift
  (`trace_pixel()`); checked against sinc and diffraction MTFs, the
  folding of frequencies, the carrier positions, the tensor walk-off and
  the paraxial pupil image (notebook 93, the sensor)
- `lightfield`: new, tested: the light field (plenoptic) camera; a thin
  main lens, a microlens array in its image plane and a sensor matched to
  the grid of microlens images (`plenoptic_camera()`, `pixel_pitch()`), raw
  images of scenes of textured planes sampled with rays (`render_raw()`),
  sub-aperture views and flat fielding (`subaperture_views()`), the
  disparity between views against distance (`disparity()`,
  `depth_from_disparity()`), shift-and-add refocusing with Fourier shifts
  (`refocus()`), sub-pixel image registration (`view_shift()`), disparity
  maps by photo consistency (`disparity_map()`), the geometric MTF of a
  refocused image (`refocused_mtf()`) and the exit pupil images of a real
  lens behind the microlenses (`microlens_images()`); checked against the
  disk overlap, the disparity of rendered planes, slanted edge MTFs, the
  depth of field of f/(nN) and the paraxial exit pupil (notebook 95, light field camera)
- `fresnel`: Fresnel lenses, tested: a parent surface (sphere, conic,
  even asphere) collapsed into zones of constant pitch, constant depth or
  given radii, with curved or flat (conical) facets and drafted risers
  (`FresnelSurface`); ray tracing on the base plane with the parent's
  normals (ideal model) or through the real sawtooth (riser hits, shadowing
  and TIR lost or traced on as stray light, optical path with the steps);
  classification of the rays of a pupil (`facet_losses()`), exact facet
  slopes for a focus (`focusing_slopes()`) and the fit of a parent to
  slopes (`surface_from_slopes()`); kinoform depth, scalar efficiency and
  the diffractive Abbe number; PMMA dispersion
- `raytrace`: sequential ray tracing, tested: spherical, conic and even
  aspheric surfaces, lenses and mirrors, ideal (paraxial) lenses, apertures
  and vignetting, indices from constants, `materials.abbe_dispersion` or any
  function (fused silica after Malitson); decentred and tilted surfaces,
  coordinate breaks (fold mirrors), tilted image planes, fields in any
  direction, object space telecentric systems; exact intersections and
  vector Snell's law, total internal reflection; ray aiming at the real
  stop; paraxial
  marginal and chief rays, cardinal points, entrance and exit pupils (system
  matrix identical to `abcd`, `to_abcd()`), Seidel sums, the Lagrange
  invariant behind every surface (`lagrange_invariants()`), Abbe's sine
  condition and Herschel's condition along real rays (`sine_condition()`:
  Conrady's OSC, checked against tan^2(U'/2) of a paraboloid, the focal
  radii of an ellipsoid, the aplanatic sphere and traced sagittal coma,
  notebook 102, sine and Herschel conditions); OPD maps over the
  pupil relative to the reference sphere on the chief ray, Fringe Zernike
  fits, RMS OPD and Strehl ratio; PSF and MTF via `imaging`; spot diagrams,
  ray fans and layout plots; example: Caldwell's all fused silica KrF
  lithography projection lens (NA 0.56, 4x) with decentre and axial shift
  sensitivities of its elements (`with_axial_shift()`), a water immersion
  ArF lens and a KrF REMA objective and zoom-axicon from patents; cones
  (`Axicon`), surfaces with a Fringe Zernike departure (`ZernikeSurface`:
  figure errors, mount deformations, freeforms), x-z sections in `draw()`; polarization ray
  trace with energy (`System.trace_polarization()`, Chipman's 3x3 matrices):
  Fresnel refraction and reflection through thin film coatings on any
  surface (`coatings.Coating`, `coatings.Layer`) with layer thickness
  variations across the surface (radial, xy or Zernike polynomials),
  coated metal mirrors, absorbing (`materials.Isotropic`, n + ik) and
  birefringent or dichroic media (`materials.Uniaxial`,
  `materials.CubicCrystal`); power along every ray, transmittance, output
  fields and Jones matrices; vectorial focus of the traced rays
  (`System.vectorial_focus()`, Debye-Wolf integral with the coatings and the
  OPD)
- `delano`: Delano's y-ybar diagram, tested: the diagram of any centred
  `raytrace.System` (powers, spacings, magnification, NA, pupils and
  cardinal points agree with the paraxial trace and `abcd`), thin lens
  systems from a drawn polygon, thick singlets with the same first order
  properties (principal planes, `thicken()`), `refine()` for hand drawn
  layouts (Petzval sum, spaces, beam heights, power signs)
- `lens_optimization`: damped least squares lens optimization, tested
  (singlet bending for minimum spherical aberration, aspheres, solves):
  curvatures, thicknesses, object distance and even aspheric terms as
  variables, solves for total length and object side NA, operands for
  transverse ray errors, distortion and telecentricity (also of the real
  chief ray through the stop centre), paraxial targets and clearances (centre and edge thicknesses at the rim of the full beam),
  stages of growing aperture and field;
  `DampedLeastSquares`, the classical lens design optimizer (Levenberg-Marquardt
  with Nielsen's damping update, Jacobian from the tolerance sensitivities):
  variables for curvatures, thicknesses, air spaces (lens shifts), conics,
  aspheric terms and model glasses (n_d, V_d) with a glass map constraint, a
  small catalogue and snapping to the nearest real glass; solves for the
  paraxial focus, the entrance pupil and a curvature for the focal length;
  operands for polychromatic ray errors, RMS spots, wavefront (OPD), Seidel
  sums and primary colour; constraints (focal length, colour, edge and centre
  thickness, length) held with Lagrange multipliers (notebook 58, lens optimization); solves for
  the length and the object side NA and the object distance as a variable,
  as in `LensOptimizer` (notebook 43, Delano diagrams, section 8: clearances as constraints
  and their prices); `predicted_gain()` ranks candidate variables (e.g. an
  asphere on each surface) by the linearized decrease of the merit function
  (notebook 43, Delano diagrams, section 7)
- `tolerancing`: Monte Carlo tolerancing, tested: tolerances (power in
  fringes, radius, thickness and air space, axial position of an element,
  image plane, index, Abbe number, element decentre) with Gaussian, uniform
  or end point distributions, applied before or after the compensators;
  performance vectors of the polychromatic RMS spot and the RMS wavefront per
  field (full pupil, valid for decentred lenses); compensators from
  `lens_optimization` variables (focus, axial manipulators) and lateral
  manipulators; sensitivity tables, variance budgets, full Monte Carlo runs and
  the linear model (exact quadratic merit), compensator choice (notebook 70, Monte Carlo tolerancing)
- `qaspheres`: new, tested: Forbes' Q-type aspheres; the Q-bfs
  polynomials (orthonormal in slope, built by Gram–Schmidt with
  Gauss–Chebyshev quadrature) and the Q-con polynomials (Jacobi
  P_m^(0,4)), their sags, `QbfsSurface` and `QconSurface` for the ray
  tracer (Newton intersection, normals, paraxial curvature and fourth
  order term for the Seidel sums), the best fit sphere through vertex and
  rim, fits of any sag (`fit_qbfs()` by slope projection, stable under
  truncation, and `fit_qcon()`), the rms slope as a manufacturability
  measure and `QCoefficient`, a variable for
  `lens_optimization.DampedLeastSquares`; checked against Forbes' closed
  forms, numerical slopes and even aspheres
  (notebook 128, Forbes Q aspheres)
- `zoom`: new, tested: mechanically compensated zoom lenses; thin lens
  conjugates, the two-group zoom (spacing and back focus for a focal
  length), the four-group zoom with variator and compensator (afocal or
  finite, both branches of the compensator), `ZoomLens` for a traced
  system with variable gaps (spacings interpolated in log f, solved
  spacings, refocusing, group positions and focal lengths) and the Olympus
  4.7× compact zoom of US 7,583,450 with catalogue glasses
  (`olympus_zoom_us7583450()`, also with its inner focusing for near
  objects) (notebook 127, zoom lenses)
- `spectacles`: new, tested: spectacle lenses and presbyopia; Hofstetter's
  fits of Duane's amplitude of accommodation and the add for a reading
  distance, the add profile along the corridor, progressive surfaces by
  sparse least squares on the second derivatives (mean power and
  astigmatism weights, hard and soft designs), the surface power and
  cylinder from the fundamental forms, sphero-cylinders from power
  matrices, and `ProgressiveLens`, a lens traced as worn from the centre
  of rotation of the eye (power and prism along every line of sight)
  (notebook 129, progressive spectacle lenses)
- `beam_quality`: beam widths, azimuth and M² after ISO 11146 (second
  moments, iterated integration window, caustic fit), tested
- `modes`: mode analysis and fibres, tested: decomposition of any field into
  Hermite-Gaussian or Laguerre-Gaussian modes (given or fitted waist,
  curvature and centre; power fractions, reconstruction error, propagation
  of the expansion), M² from the mode content (exact second moments of
  coherent superpositions); LP modes of weakly guiding step-index fibres
  (characteristic equation, cutoffs, b(V), n_eff, mode fields, Marcuse's
  Gaussian approximation); coupling efficiency of any field into any mode
  (generalizes `abcd.coupling_efficiency()`), sweeps over offset, tilt and
  defocus, power per fibre mode
- `vortices`: new, tested: optical vortices and orbital angular momentum of
  scalar beams; spiral phase plates (continuous, multilevel, fractional
  charge) and their height profiles, fork gratings (binary and sinusoidal
  amplitude, blazed and binary phase); the topological charge on a circle,
  all phase singularities of a sampled field with their signs, the spiral
  (OAM) spectrum about any axis, the mean OAM per photon and the orbital
  current density; checked against LG modes, the sinc² spectrum of
  multilevel plates, the spectrum of fractional plates, the analytic
  splitting of a charge-3 vortex and the order efficiencies of gratings
  (notebook 69, optical vortices)
- `ultrafast`: new, tested: ultrashort pulses through glass, lenses,
  prisms and gratings; group index, GVD and TOD of any dispersion formula
  (fused silica and CaF2 after Malitson), the exact spectral phase of a
  slab, Gaussian pulse broadening and the best input duration, pulse
  synthesis on free time grids; Bor's pulse-front delay from the chromatic
  power derivative and phase delay, group delay and GDD summed along rays
  of a `raytrace.System`, the focal field spectrum of a traced lens (sums of
  monochromatic focuses in time) and Bor and Horváth's axial box model;
  pulse-front tilt from angular dispersion (prisms and gratings in closed
  form, group delays of rays through a prism, x-t synthesis of a dispersed
  beam); exact focused Gaussian beamlets and the axial pulse of
  simultaneous spatial and temporal focusing (notebook 81, ultrashort pulses)
- `fibres`: new, tested: optical fibres in practical units (µm, km, ps,
  ps/(nm km)); radial index profiles (step, graded with any exponent,
  triangular, W, GeO₂-doped silica after Fleming and a G.652 standard fibre
  model `smf28()`), the NA, V number and acceptance angle; ray classes
  (bound, tunnelling, refracting), meridional and skew rays in a step core,
  graded cores as `grin.IndexField` for `grin.trace()`; a radial finite
  volume scalar mode solver (LP modes of any profile, checked against the
  characteristic equation and the parabolic analytic modes), exact vector
  modes of the step fibre, Petermann II mode field diameter and Marcuse's
  bend loss; effective and group index, dispersion D and slope by finite
  differences, material and waveguide dispersion (Gloge), WKB group delays
  and delay spreads of multimode fibres and the optimal profile exponent
  (Olshansky and Keck), PMD of random birefringent sections; loss spectrum
  of silica (Rayleigh, UV, IR and OH); fibre Bragg gratings by coupled mode
  theory (uniform analytic, piecewise uniform transfer matrices with
  apodization, chirp and phase shifts; checked against `thin_films`), group
  delay and the Bragg shift with strain and temperature (notebooks 115 (light in a fibre) to
  120 (fibre link))
- `fibrecomm`: new, tested: a basic intensity modulated fibre link; PRBS
  patterns, NRZ transmitters with extinction ratio, rise time and transient
  chirp, linear pulse propagation with β₂, β₃ and loss (checked against
  the Gaussian formulas), PIN and APD receivers with shot and thermal noise
  and a Gaussian filter, Q factor, BER and sensitivity (Agrawal), optimum
  APD gain, eye diagrams and openings, power and rise-time budgets and the
  Bellcore dispersion penalty (notebooks 118 (pulse broadening) to 120 (fibre link))
- `TIE`: basic transport of intensity solver, partly tested
- `ifta`: basic Gerchberg-Saxton and variants for beam shaping, tested
  (convergence of GS, HIO-HOO and over-compensation on a reachable target,
  phase quantization)
- `wavefront_shaping`: new, tested: focusing through scattering media;
  random phase screens, SLM segments, transmission matrices of any linear
  propagator (segment by segment), phase conjugation for full, phase-only,
  P-level and binary amplitude modulation with the enhancement and its
  theoretical value, Vellekoop's feedback algorithms (stepwise and
  continuous sequential, random partitioning), the phase-stepping TM
  measurement against a reference (canonical or Hadamard basis), a
  weighted Gerchberg-Saxton loop through the TM (built on `ifta`), random
  unitaries and the Marchenko-Pastur law (notebook 73, wavefront shaping)
- `lensless`: new, tested: lensless imaging; the pinhole camera
  (Rayleigh's optimum pinhole, the exact on-axis Fresnel intensity behind a
  hole, geometric and diffraction blur), coded apertures (MURAs of prime
  order 4m + 1 and their decoders, the periodic shadow model, decoding by
  correlation, Fenimore's signal-to-noise ratio against a single hole), and
  the diffuser camera (PSF of a point source behind any mask by wave
  propagation, the focal length of the bumps of a random phase plate, the
  shift and 1 + z/u scaling of the PSF, Wiener deconvolution and ADMM with
  total variation, positivity and a cropped sensor) (notebook 79, lensless imaging)
- `phase_retrieval`: phase retrieval, tested: Fienup's error reduction and
  hybrid input-output with support, reality and positivity constraints,
  shrink-wrap support and the autocorrelation support, error histories,
  removal of the trivial ambiguities (shift, constant phase, twin image);
  pupil phase retrieval from through-focus PSFs (phase diversity) by a
  multi-plane Gerchberg-Saxton iteration (with unwrapping via `unwrap`) or a
  least squares fit of Fringe Zernikes with the analytic Jacobian, with
  paraxial (Z4) or exact (angular spectrum) defocus; near-field (Fresnel)
  coherent diffractive imaging with any pyoptics propagator in place of the
  Fourier transform
- `shack_hartmann`: Shack-Hartmann wavefront sensor, tested: spot images
  (Fraunhofer pattern of each lenslet on its camera window) from a pupil
  phase, e.g. a ray-traced `raytrace.Pupil`, thresholded centroids with
  optional reference image, intensity weighted model slopes, modal
  reconstruction (least squares fit of Fringe Zernike gradients) and zonal
  reconstruction (Southwell); sensitivity and dynamic range of the design
- `turbulence`: new, tested: atmospheric turbulence; von Kármán phase
  spectrum, random phase screens (FFT method with subharmonics and cell
  averaged low frequencies), measured and theoretical structure functions,
  Noll's Zernike variances and residuals; Hufnagel-Valley profiles cut into
  layers, r0, isoplanatic angle, coherence time and the Rytov scintillation
  index with aperture averaging; long and short exposure transfer functions,
  seeing with outer scale, Fried's lucky imaging probability; split-step
  propagation through layers and frozen flow screens; checked against the
  Kolmogorov structure function, Noll's table, the 19.12 λ^(-7/6) formula and
  a split-step simulation (notebook 72, twinkling stars and AO)
- `coronagraph`: new, tested: stellar coronagraphs in the four planes
  pupil, focal plane mask, Lyot stop and camera; telescope pupils with
  obscuration and spiders (`telescope_pupil()`), the matrix Fourier
  transform (`mft()`), Sonine and circular prolate apodizers
  (`sonine_apodizer()`, `sonine_psf()`, `prolate_apodizer()`), occulting
  disks by Babinet's principle (`occulter_lyot_field()`,
  `occulter_lyot_analytic()`), vortex masks of any charge with nested,
  smoothly windowed focal grids (`vortex_lyot_field()`,
  `vortex_lyot_analytic()`, `vortex_blocks()`), the `Coronagraph` class
  with contrast normalization, power law aberration screens, deformable
  mirror influence functions and electric field conjugation
  (`efc_jacobian()`, `efc_step()`); checked against the Airy and Sonine
  PSFs, J_0(pi a), the prolate eigenvalue, the analytic vortex fields,
  the tip-tilt leak laws and the Zernike selection rule (notebook 96, coronagraphs)
- `telescopes`: new, tested: reflecting telescopes as `raytrace` systems:
  Newtonian (`newtonian()`), classical Cassegrain and Ritchey–Chrétien
  from primary focal length, system focal length and back focus
  (`two_mirror_telescope()`, `two_mirror_geometry()`, Schwarzschild's
  conics `rc_conics()`, `cassegrain_conic()`), the Schmidt camera with a
  corrector plate refined on the traced wavefront (`schmidt_camera()`,
  `schmidt_plate()`); closed form third order results
  (`mirror_aberrations()`, `two_mirror_aberrations()`,
  `schmidt_chromatic_blur()`), spot diagrams over an obscured pupil
  (`spot()`) and best focus (`best_focus()`); checked against the Seidel
  sums, the coma and astigmatism formulas, the medial image surface and
  the chromatic blur of the plate (notebook 97, telescope zoo)
- `segmented`: new, tested: segmented telescope apertures: hexagonal
  segments in rings (`hex_centres()`, `hexagon_vertices()`,
  `segment_masks()`), antialiased convex polygons and struts
  (`polygon_mask()`, `strut_mask()`), a JWST-like aperture
  (`jwst_like_pupil()`), the closed form Fourier transform of a polygon
  (`polygon_ft()`) and the array factor, pupil fields with segment piston
  and tip-tilt (`segment_field()`) and their far fields by MFT
  (`far_field()`); coarse phasing with a dispersed fringe sensor
  (`dispersed_fringes()` with finite spectral channels, `dfs_piston()`)
  and fine phasing by multi-plane Gerchberg–Saxton phase retrieval from
  defocused images (`diversity_psfs()`, `misell()`, `fit_segments()`);
  checked against the polygon transform, the phasor sum Strehl ratio, the
  random piston halo, sinc(p dsigma) and the unique range 1/(2 dsigma)
  (notebook 98, segmented mirrors)
- `lensing`: new, tested: gravitational lensing: light bending in the weak
  field index n = 1 + R_s/r and in the exact Schwarzschild metric as a
  medium for `grin.trace()` (`weak_field_medium()`,
  `schwarzschild_medium()`), the exact deflection integral
  (`schwarzschild_deflection()`), Schwarzschild and Einstein radii; thin
  lenses (`PointMass`, `SIS`, `ExternalShear`, `Composite`) with the lens
  map, Jacobian and magnification, image finding by Newton iteration
  (`find_images()`), critical curves and caustics, the Fermat potential
  and inverse ray shooting of extended sources (`ray_shoot()`); checked
  against the point mass images and the Paczynski curve, the Witt–Mao
  disk magnification, the SIS images and time delay and the cusps of the
  SIS plus shear caustic (notebook 99, Einstein rings)
- `echelle`: new, tested: echelle spectrographs: the grating equation
  with the off-plane angle, Littrow orders, blaze wavelengths, the free
  spectral range and the scalar blaze function (`Echelle`), grating and
  prism cross-dispersers, positions in the echellogram and its image for
  a spectrum (`order_position()`, `echellogram()`), the staircase
  wavefront of the facets and its far field by FFT
  (`staircase_far_field()`, `order_efficiencies()`), the instrument
  profile of a slit and a diffraction limited beam and the slit limited
  resolving power (`instrument_profile()`, `slit_resolving_power()`),
  and a sunlike spectrum with Fraunhofer lines; checked against
  sinc^2(m - m_B), R = m N, 2 tan(theta_B) f/s, the order lengths
  2 f tan(theta_B)/m and the lambda^2 order spacing of a grating
  cross-disperser (notebook 100, echelle spectrograph)
- `adc`: new, tested: atmospheric refraction and dispersion and the
  atmospheric dispersion corrector: the index of air after Edlén (Birch
  and Downs) with temperature, pressure and humidity (`edlen_index()`),
  refraction in plane layers, the A tan z - B tan^3 z formula, the exact
  integral of a spherical isothermal atmosphere and a ray trace through it
  as a GRIN medium (`refraction_integral()`, `atmosphere_medium()`,
  `trace_refraction()`), the dispersion against the zenith angle
  (`atmospheric_dispersion()`), Amici prisms (thin and exact,
  `AmiciPrism`), two counter-rotating prisms (`adc_dispersion()`,
  `adc_residual()`), the search of a glass catalogue for pairs matched to
  the dispersion of air (`best_glass_pairs()`) and the image of a
  dispersed star (`smeared_star()`); checked against Ciddor's formula,
  the exact refraction integral and the thin prism formula (notebook 101, atmospheric dispersion corrector)
- `arvr`: new, tested: AR/VR and head-up display optics. Waveguide
  combiners in k-space: field angles and transverse wave vectors
  (`field_to_k()`, `k_to_field()`), the vector grating equation
  (`diffract()`), guiding by total internal reflection (`is_guided()`),
  the largest 1D field of view against the index (`max_fov_1d()`) and the
  symmetric period; pupil replication: bounce spacing, out-coupling
  efficiencies for equal copies (`uniform_outcoupling()`), the powers of
  the copies and the eye box efficiency of an eye pupil
  (`eyebox_efficiency()`); layers for `litho.rcwa` of binary, slanted and
  volume gratings (`binary_grating()`, `slanted_grating()`,
  `volume_grating()`); Kogelnik's coupled wave theory of thick
  transmission and reflection gratings (`kogelnik()`, `bragg_angle()`);
  crystal quartz after Ghosh, the exact retardance of an A-plate at any
  angle of incidence (`aplate_retardance()`), Pancharatnam's achromatic
  quarter wave plate and the Jones model of a pancake lens with its
  direct and five-pass ghosts (`pancake_jones()`); the double image of a
  windshield and the wedge that removes it (`windshield_double_image()`,
  `wedge_for_distance()`), a ray traced HUD with a freeform mirror
  (`hud_system()`) and the vergence matrix seen by the eye (`aim_ray()`,
  `vergence()`); checked against RCWA, thin film matrices, tanh² at Bragg
  and the unfolded distance of plane mirrors (notebooks
  123 (AR waveguide combiners), 124 (volume holographic gratings),
  125 (pancake lenses) and 126 (head-up displays))
- `lidar`: new, tested: LiDAR link budgets and scanners. The echo of a
  Lambertian target (`lambertian_return()`), Kim's extinction of haze and
  fog (`extinction_coefficient()`), a model of the direct solar spectrum
  and the background in a receiver (`solar_irradiance()`,
  `background_power()`); APD SNR with excess noise (`apd_snr()`),
  detection probability and Rice's false alarm rate; Coates' first-photon
  histogram of a SPAD and its correction (`coates_histogram()`,
  `coates_correction()`); class 1 eye safety after IEC 60825-1:2014 for
  700 to 1400 and 1500 to 1800 nm (`ael_class1()`, C4, C7, measurement
  apertures, the three pulse train rules in `class1_pulse_energy()`);
  FMCW beat frequencies, range and velocity, a simulated beat with laser
  phase noise and nonlinear chirps (`fmcw_beat_signal()`), the coherent
  fraction and the SNR of a balanced coherent receiver
  (`coherent_snr()`); resolvable spots, an exact XY galvo
  (`galvo_xy()`), a polygon mirror (`polygon_scan()`), Lissajous paths,
  the array factor of an optical phased array with quantized and noisy
  phases (`opa_array_factor()`), grating emitter angles and an exact
  Risley prism pair (`risley_direction()`); checked against Monte Carlo
  of Rice's rate, first photons and phase noise, the shot noise limit,
  Ruze's formula and first order prism deviations (notebooks
  130 (time-of-flight LiDAR), 131 (FMCW LiDAR) and 132 (beam scanning))
- `structured_light`: new, tested: triangulation and structured light.
  Pinhole cameras and projectors (`Pinhole`, `look_at()`), rays against
  depth map scenes by fixed point iteration or marching with bisection
  (`intersect_depth()`); laser triangulation depth, sensitivity and the
  speckle limit; phase shifted fringe patterns with projector gamma,
  temporal phase unwrapping and triangulation against planes of light
  (`fringe_patterns()`, `temporal_unwrap()`, `triangulate_column()`);
  fan-out gratings by IFTA with quantization and a pixel search
  (`fanout_ifta()`, `order_efficiencies()`), the dots of a VCSEL array
  and grating (`dot_directions()`, `render_dots()`), block matching with
  normalized cross correlation and depth from disparity; checked by exact
  round trips of camera, projector and phase
  (notebook 133, triangulation and structured light)
- `wpm`: wave propagation method (Cython extension) for inhomogeneous, also
  absorbing media, 2D and y-invariant 1D fields, multi-step driver; direct
  O(N^2) sum (OpenMP) or the fast "levels" variant (one FFT pair per distinct
  index), chosen automatically; tested against Rayleigh-Sommerfeld, plane
  waves (tilted, evanescent, absorbed), slabs and a GRIN mode
- `processing`: image processors for single images and image stacks, tested:
  `Scaler` (peak, mean or total of field strength or intensity),
  `CameraNoiser` (Gaussian, Poisson shot noise, linear camera model after
  EMVA 1288 with dark current, read noise, full well, gain, ADC and fixed
  pattern noise), `Shifter` (subpixel Fourier shift) and `Centerer`
  (centering on the intensity centroid, `utils.center_of_mass()`)
- `sketch`: schematic drawings of optical layouts (thin lenses, stops,
  prisms, mirrors, detectors) with paraxially traced rays, tested: thin lens
  ray tracing in both directions; ready-made sketches of a Köhler
  microscope, Abbe's 4f picture with filtered orders, phase contrast, DIC,
  confocal and LED array (Fourier ptychography) microscopes; primitives for
  folded layouts (beam splitters, rotated lenses and detectors, spherical
  surfaces, lenslet arrays, beams as bands between edge rays) and sketches
  of a Twyman-Green and a Fizeau interferometer, a Shack-Hartmann and a
  phase diversity lens test, off-axis holography (lensless and microscope),
  a coated fold mirror with s and p, the KrF illuminator with its pupil and
  field planes, a spatial filter, the geometry of rainbows and of the
  glory seen from an aircraft, the Foucault knife-edge (and wire and Ronchi)
  test of a concave mirror, a Toepler schlieren system and a direct
  shadowgraph, pulses through lenses, prisms and gratings (pulse-front
  delay and tilt), and the formation and reduction of speckle (polarization,
  wavelength and moving diffuser diversity, a laser projector), LEDs above a
  plane and LED collimators, the ray map of a beam shaper, a two-lens
  flat-top shaper and a freeform lens projecting a picture; fibres (step,
  graded and single-mode cross sections, the acceptance cone, skew rays in
  an end view, the transverse resonance of a slab, the dispersion
  mechanisms, a fibre Bragg grating and a basic link); AR/VR and head-up
  displays (a waveguide combiner from the side and with a fold grating, a
  volume grating at Bragg, a pancake lens, a windshield HUD and its double
  image); 3D sensing (time-of-flight and FMCW LiDARs, galvo, polygon,
  MEMS, phased array and Risley scanners, laser triangulation, fringe
  projection and a dot projector)
- `interpolator`: trigonometric (Fourier) interpolation of periodic images
  and complex fields, tested: along one axis at any points
  (`trig_interpolate()`), in 2D on any grid or at scattered points
  (`trig_interpolate_2d()`), refinement by zero padding the spectrum
  (`fourier_upsample()`), `TrigInterpolator` with a frequency cutoff (e.g.
  2NA/wavelength), exact level crossings of a periodic cut
  (`level_crossings()`, used by `litho.metrics.threshold_cd(...,
  interpolation="fourier")`); windows against the Gibbs phenomenon
  (`smoothing_window()`: Lanczos sigma factors of any order, Fejér, Hann,
  Gaussian); exact for band-limited data, checked against the Gibbs
  overshoot of a step (notebook 107, Fourier interpolation of aerial images)
- Unfinished stubs: the functions in `polarization._polarization` are empty

## Tests

The core is tested: a suite of more than 2000 pytest tests checks the numerics
against analytical results (Fresnel and Fraunhofer diffraction, Gaussian beam
propagation, Mahajan's on-axis intensity, Airy pattern and OTF, Fresnel
equations, thin film stacks, Goos-Hänchen shift, beam widths and M² of
Hermite-Gaussian beams, Mueller and Jones calculus, FFT conventions,
Richards-Wolf integrals for high NA foci, two-beam interference of TE and
TM waves, in air and in a resist, standing waves in a resist on silicon,
threshold CDs and process windows of analytic images, apodized two-beam
images, ABCD matrices against lens and resonator formulas and the Collins
integral against Gaussian beams, fields on tilted planes against plane
waves and Gaussian beams at the plane's 3D points, power flux through
tilted planes, phase unwrapping of wrapped Zernike
wavefronts, vortices and noisy wavefronts, phase-shifting and Fourier
fringe analysis of known Zernike phases, holograms of known objects
reconstructed, refocused and autofocused, exact ray traces against Snell's
law, perfect conics, Seidel sums and the Airy pattern, tilted plates and
fold mirrors, Fienup's HIO and phase diversity recovering known objects and
Zernike pupil phases, HG/LG mode decompositions and M² from the mode content
against ISO caustic fits, LP fibre modes against cutoffs and the Marcuse
mode field radius, fibre coupling against the Gaussian formulas) and runs in GitHub
Actions on Python 3.9, 3.11 and 3.13 for every push and pull request. A
consistent phase convention is used throughout (see {doc}`conventions`).
