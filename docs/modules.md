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
  inclined planes (detectors); `tilted_plane()` and
  `vectorial_tilted_plane()` are also available from `propagators`
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
- `fft`, `utils`, `masks`, `geometry`, `analytical`: mature and tested
  (`fft.czt()`: Bluestein's chirp-z transform; `fft.zoom_FT()` and
  `zoom_inv_FT()` evaluate `FT()` and `inv_FT()` on any equidistant grid)
  (`utils.pad()`/`unpad()` for centered zero padding; `analytical` holds the
  Airy pattern, its encircled energy, the diffraction limited OTF and the
  exact series solutions for a dielectric cylinder (TE) and sphere (scalar),
  also for beams: `cylinder_field()`, `sphere_field()`, and the vector
  Lorenz-Mie solution of a sphere: `mie_coefficients()`, `mie_field()`,
  `mie_efficiencies()`)
- `bases`: mature (Zernike, Legendre, Chebyshev bases, numerical
  orthogonalization, fits); all bases can be shifted and rotated, Legendre and
  Chebyshev derive from the abstract `TensorProductBasis`; tested
- `beams`: tested (plane, spherical, Hermite-Gaussian and Laguerre-Gaussian
  beams)
- `interfaces`, `thin_films`: mature and tested (Fresnel coefficients,
  coherent multilayer stacks; `stack_transfer_functions()` gives r and t on
  the angular spectrum grid of a field)
- `materials`: Abbe (n_d, V_d) dispersion model via a two-term Cauchy
  formula, tested
- `glasses`: optical glass catalogues of SCHOTT, OHARA, HOYA, CDGM, SUMITA
  and HIKARI (about 1600 glasses with the vendors' Sellmeier and power
  series formulas, from the refractiveindex.info database, CC0), tested:
  a `Glass` is a material n(wavelength_nm) for `raytrace.Surface`, Abbe
  numbers and relative partial dispersions for any lines, the normal line
  and dP_gF, catalogue search by n_d/V_d (and dP_gF), filters by status,
  index and Abbe number, secondary spectrum of thin achromats; real glasses
  for `lens_optimization.DampedLeastSquares.snap_glasses()` (notebook 61)
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
  field); mask stacks in the thin mask approximation (binary, attenuated
  PSM), with absorber transmissions from `thin_films` (angle and TE/TM
  dependent) and an optional boundary layer model for thick mask edges;
  projector with immersion (or dry), vectorial or scalar model, Fringe
  Zernike aberrations and pupil apodization (amplitude transmission
  1 + sum of Fringe Zernikes, optionally as log amplitude, or any
  transmission function); images in the resist on a thin film wafer stack (BARC, top
  coat, substrate; TE/TM Fresnel transmission, standing waves); metrics
  (contrast, ILS/NILS, threshold CD, dose to size, MEEF, focus-exposure
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
  matrix identical to `abcd`, `to_abcd()`), Seidel sums; OPD maps over the
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
  thickness, length) held with Lagrange multipliers (notebook 58); solves for
  the length and the object side NA and the object distance as a variable,
  as in `LensOptimizer` (notebook 43, section 7: clearances as constraints
  and their prices)
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
- `TIE`: basic transport of intensity solver, partly tested
- `ifta`: basic Gerchberg-Saxton and variants for beam shaping, tested
  (convergence of GS, HIO-HOO and over-compensation on a reachable target,
  phase quantization)
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
  field planes, and a spatial filter
- Unfinished stubs: `interpolator.TrigInterpolator` cannot be evaluated yet,
  the functions in `polarization._polarization` are empty

## Tests

The core is tested: a suite of about 900 pytest tests checks the numerics
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
