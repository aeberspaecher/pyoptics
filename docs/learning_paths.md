# Learning paths

The {doc}`examples` are numbered by topic and in the order they were
written, not in the order in which they are best read. This page suggests
paths through them for different readers. Each path has a **core** to read
in order, followed by optional side trips. The notebooks stand on their own,
so leaving a path or mixing two is fine; where a notebook builds on an
earlier one, its introduction usually says so.

Most readers profit from {doc}`conventions` and {doc}`00 Foundations and conventions <notebooks/00_foundations_conventions>` early: they fix the
signs of phases, radii, angles and Zernikes that every other notebook uses.
Readers new to optics can leave notebook 00 for later and read it as a
summary.

A few notebooks serve almost every path:

- {doc}`52 Abbe image formation <notebooks/52_abbe_image_formation>` is the entry to all of imaging, despite its high number.
- {doc}`17 PSF, MTF, Strehl <notebooks/17_psf_mtf_strehl>` defines the image quality measures that the later notebooks report.
- {doc}`47 Rays and waves <notebooks/47_rays_and_waves>`, {doc}`21 Lens focus from rays to waves <notebooks/21_lens_focus_waves>` and {doc}`56 Gaussian beam decomposition <notebooks/56_gaussian_beam_decomposition>` connect ray tracing with wave optics.

## Undergraduate in a first optics course

For readers who know the textbook theory (Hecht level) and want to see it
work. The core follows the order of a typical lecture.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`01 Free-space propagation <notebooks/01_propagation>` | Slits, apertures, Talbot effect and the Poisson spot: diffraction you can check by hand. |
| 2 | {doc}`06 Gaussian beams <notebooks/06_gaussian_beams>` | Gaussian beams as the paraxial building block, and the Gouy phase. |
| 3 | {doc}`08 ABCD matrices <notebooks/08_abcd_matrices>` | Paraxial rays and beams through lenses and resonators with 2×2 matrices. |
| 4 | {doc}`10 Interfaces and thin films <notebooks/10_interfaces_thin_films>` | Fresnel coefficients, Brewster angle, total internal reflection, soap films. |
| 5 | {doc}`11 Jones and Mueller calculus <notebooks/11_polarization>` | Polarization with Jones vectors and Stokes parameters. |
| 6 | {doc}`52 Abbe image formation <notebooks/52_abbe_image_formation>` | Why a microscope has a resolution limit: diffraction orders and the pupil. |
| 7 | {doc}`15 Imaging and Zernikes <notebooks/15_imaging_zernikes>` | Resolution and aberrations described by Zernike polynomials. |
| 8 | {doc}`17 PSF, MTF, Strehl <notebooks/17_psf_mtf_strehl>` | PSF, MTF and Strehl ratio, the language of image quality. |
| 9 | {doc}`23 Interferometry <notebooks/23_interferometry>` | How interferometers turn fringes into wavefronts. |
| 10 | {doc}`74 Coherence and stellar interferometry <notebooks/74_coherence_stellar_interferometry>` | Spatial and temporal coherence, and how they measure stars. |

**Side trips:** {doc}`71 Rainbows and glories <notebooks/71_rainbows_and_glories>` (Descartes' ray picture and Airy's wave picture of
the same bow), {doc}`47 Rays and waves <notebooks/47_rays_and_waves>` (a lab rule explained with rays and waves), {doc}`68 Bessel and Airy beams <notebooks/68_bessel_airy_self_healing>`
(beams that seem not to diffract). Afterwards, {doc}`00 Foundations and conventions <notebooks/00_foundations_conventions>` shows where everything
fits on the ladder from Maxwell's equations to rays.

## Curious reader

For readers who want the story behind a phenomenon and are happy to read the
figures without running the code. No order is needed.

| Notebook | Story |
|---|---|
| {doc}`71 Rainbows and glories <notebooks/71_rainbows_and_glories>` | Why the rainbow is where it is, and the glory around an aircraft's shadow. |
| {doc}`72 Twinkling stars and AO <notebooks/72_twinkling_stars_adaptive_optics>` | Why stars twinkle, and how telescopes undo it. |
| {doc}`76 Schlieren and knife-edge <notebooks/76_schlieren_knife_edge>` | Foucault's razor blade test and seeing air flow. |
| {doc}`51 Fresnel lenses <notebooks/51_fresnel_lenses>` | From lighthouse lenses to solar concentrators. |
| {doc}`47 Rays and waves <notebooks/47_rays_and_waves>` | Which way round a plano-convex lens should go. |
| {doc}`69 Optical vortices <notebooks/69_optical_vortices>` | Light with a twisted wavefront. |
| {doc}`68 Bessel and Airy beams <notebooks/68_bessel_airy_self_healing>` | Beams that heal behind an obstacle. |
| {doc}`79 Lensless imaging <notebooks/79_lensless_coded_aperture_imaging>` | Cameras without lenses: pinholes, masks and diffusers. |
| {doc}`87 Speckle reduction <notebooks/87_speckle_reduction>` | Why a laser spot glitters, and how to stop it. |
| {doc}`83 Étendue <notebooks/83_etendue_brightness_theorem>` | Why a burning glass cannot make anything hotter than the Sun. |
| {doc}`80 Carpet cloak <notebooks/80_carpet_cloak>` | Hiding a bump under a carpet with a graded material. |
| {doc}`90 Bokeh <notebooks/90_bokeh>` | Why the background of a fast lens shows cat's eyes, soap bubbles and onion rings. |
| {doc}`91 Sunstars <notebooks/91_sunstars>` | Why a stopped down lens turns the Sun into a star, and how many rays it gets. |
| {doc}`92 Flare and ghosts <notebooks/92_flare_ghosts>` | Where the coloured polygons come from when the Sun is in the frame, and what lens coatings do about them. |
| {doc}`93 The sensor <notebooks/93_sensor>` | Where moiré and false colour come from, what the low pass filter in front of a sensor does, and why some lenses darken the corners on digital cameras. |
| {doc}`94 Depth of field and tilt-shift <notebooks/94_tilt_shift_depth_of_field>` | How much is sharp, when stopping down stops helping, and how a tilted lens puts a whole landscape in focus at f/2.8. |
| {doc}`95 The light field camera <notebooks/95_light_field_camera>` | How a camera with microlenses in front of the sensor can be focused after the picture was taken, and what it pays for that. |
| {doc}`96 Coronagraphs <notebooks/96_coronagraphs>` | How a telescope can see a planet ten billion times fainter than the star next to it. |
| {doc}`99 Einstein rings <notebooks/99_einstein_rings>` | How a galaxy bends light like a lens, turning a distant quasar into a ring or a cross, and why a star passing behind another brightens. |
| {doc}`98 Segmented mirrors <notebooks/98_segmented_mirrors>` | Where the six plus two spikes of JWST images come from, and how 18 mirror segments are aligned to a few nanometres. |
| {doc}`97 Telescope zoo <notebooks/97_telescope_zoo>` | Why big telescopes are Ritchey–Chrétiens, why a Newtonian has comet-shaped stars at the edge, and how the Schmidt camera sees a wide field with a spherical mirror. |

## Lens designer

For readers who use a commercial lens design program and want to compare
pyoptics with it, or go from the ray trace to the wave optical image.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`00 Foundations and conventions <notebooks/00_foundations_conventions>` | Sign conventions for radii, angles, OPD and Zernikes. |
| 2 | {doc}`08 ABCD matrices <notebooks/08_abcd_matrices>` | Paraxial layout with ABCD matrices. |
| 3 | {doc}`43 Delano diagrams <notebooks/43_litho_delano_lens_design>` | First order layout with $y$-$\bar y$ diagrams. |
| 4 | {doc}`19 Ray tracing to OPD <notebooks/19_ray_tracing>` | Exact ray tracing, OPD maps, Zernikes over the field. |
| 5 | {doc}`61 Glass catalogues <notebooks/61_glass_catalogues>` | Glass maps, partial dispersion and an apochromat. |
| 6 | {doc}`44 Seidel aberrations <notebooks/44_seidel_caldwell_lens>` | Seidel sums surface by surface: where correction and tension come from. |
| 7 | {doc}`58 Lens optimization <notebooks/58_lens_optimization>` | Damped least squares optimization, from a singlet to a microscope objective. |
| 8 | {doc}`45 Zernike aberration gallery <notebooks/45_zernike_aberration_gallery>` | What each Zernike term looks like in a real lens. |
| 9 | {doc}`17 PSF, MTF, Strehl <notebooks/17_psf_mtf_strehl>` | PSF, MTF and Strehl from the pupil. |
| 10 | {doc}`50 Magnification, distortion, telecentricity <notebooks/50_magnification_distortion_telecentricity>` | Magnification, distortion and telecentricity as Zernikes. |
| 11 | {doc}`70 Monte Carlo tolerancing <notebooks/70_monte_carlo_tolerancing>` | Monte Carlo tolerancing. |

**From rays to waves:** {doc}`21 Lens focus from rays to waves <notebooks/21_lens_focus_waves>`, {doc}`47 Rays and waves <notebooks/47_rays_and_waves>`, {doc}`46 Analytic phase propagation <notebooks/46_analytic_phase_propagation>` (propagating strongly curved
wavefronts), {doc}`56 Gaussian beam decomposition <notebooks/56_gaussian_beam_decomposition>`.
**Polarization and coatings:** {doc}`20 Polarization ray tracing <notebooks/20_polarization_raytrace>`, {doc}`14 Coated laser mirror <notebooks/14_coated_laser_mirror>`.
**Broadband light:** {doc}`81 Ultrashort pulses <notebooks/81_ultrashort_pulses_lenses>`.
**Camera lenses:** {doc}`90 Bokeh <notebooks/90_bokeh>` (a double Gauss, vignetting, out-of-focus rendering), {doc}`91 Sunstars <notebooks/91_sunstars>` (diffraction at the iris), {doc}`92 Flare and ghosts <notebooks/92_flare_ghosts>` (double reflections, coatings), {doc}`93 The sensor <notebooks/93_sensor>` (pixels, Bayer mosaic, low pass filter, microlenses), {doc}`94 Depth of field and tilt-shift <notebooks/94_tilt_shift_depth_of_field>` (hyperfocal distance, optimum aperture, Scheimpflug), {doc}`95 The light field camera <notebooks/95_light_field_camera>` (microlenses, views, refocusing).

## Lithography and semiconductor engineer

For readers who work with scanners, masks or process windows. The core
builds up an imaging simulation one part at a time.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`52 Abbe image formation <notebooks/52_abbe_image_formation>` | Abbe's method, partial coherence and the pupil cutoff. |
| 2 | {doc}`17 PSF, MTF, Strehl <notebooks/17_psf_mtf_strehl>` | Image quality measures used throughout. |
| 3 | {doc}`31 Vectorial imaging in lithography <notebooks/31_vectorial_imaging_lithography>` | Hyper-NA imaging: s, p and unpolarized lines and spaces. |
| 4 | {doc}`32 Lithography toolbox <notebooks/32_lithography_toolbox>` | The litho toolbox: sources, masks, stacks, process windows. |
| 5 | {doc}`34 Hopkins imaging and SOCS <notebooks/34_hopkins_socs>` | Hopkins imaging, TCCs and SOCS for fast mask iterations. |
| 6 | {doc}`35 Resist and swing curves <notebooks/35_litho_resist_swing>` | A chemically amplified resist and swing curves. |
| 7 | {doc}`36 Rigorous mask (RCWA) <notebooks/36_litho_rigorous_mask>` | Rigorous mask diffraction with RCWA, mask 3D effects. |
| 8 | {doc}`37 Mask models compared <notebooks/37_litho_mask_models>` | Kirchhoff, film, boundary layer and RCWA side by side. |
| 9 | {doc}`65 FDTD mask for lithography <notebooks/65_litho_fdtd_mask>` | An FDTD mask for lines and contact holes. |
| 10 | {doc}`38 Jones pupils <notebooks/38_litho_jones_pupil>` | Polarization of the projection lens as a Jones pupil. |

**The projection lens and the scanner:** {doc}`21 Lens focus from rays to waves <notebooks/21_lens_focus_waves>`, {doc}`33 Lens aberrations across the slit <notebooks/33_litho_lens_aberrations>`, {doc}`41 Telecentricity across the slit <notebooks/41_litho_telecentricity>`, {doc}`50 Magnification, distortion, telecentricity <notebooks/50_magnification_distortion_telecentricity>`
(overlay), {doc}`43 Delano diagrams <notebooks/43_litho_delano_lens_design>`, {doc}`44 Seidel aberrations <notebooks/44_seidel_caldwell_lens>`, {doc}`70 Monte Carlo tolerancing <notebooks/70_monte_carlo_tolerancing>`, {doc}`42 KrF illuminator <notebooks/42_litho_illuminator>`, {doc}`86 Homogenizers <notebooks/86_homogenizers>`.
**Computational lithography:** {doc}`39 Source optimization <notebooks/39_litho_source_optimization>`, then {doc}`40 OPC <notebooks/40_litho_opc>`.

## Illumination engineer

For readers who think in étendue, irradiance and uniformity rather than in
PSFs. This group is nearly self-contained; {doc}`19 Ray tracing to OPD <notebooks/19_ray_tracing>` introduces the ray tracer
that it uses.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`83 Étendue <notebooks/83_etendue_brightness_theorem>` | Étendue, phase space and the brightness theorem. |
| 2 | {doc}`84 Nonimaging concentrators <notebooks/84_nonimaging_concentrators>` | The compound parabolic concentrator and the edge ray principle. |
| 3 | {doc}`88 LED optics <notebooks/88_led_optics_monte_carlo>` | LED optics and Monte Carlo illuminance. |
| 4 | {doc}`86 Homogenizers <notebooks/86_homogenizers>` | Fly's eye lenslet arrays and integrator rods. |
| 5 | {doc}`85 Köhler and critical illumination <notebooks/85_koehler_critical_illumination>` | Köhler and critical illumination. |
| 6 | {doc}`51 Fresnel lenses <notebooks/51_fresnel_lenses>` | Fresnel lenses and solar concentrators. |
| 7 | {doc}`42 KrF illuminator <notebooks/42_litho_illuminator>` | A complete illuminator, ray traced. |
| 8 | {doc}`87 Speckle reduction <notebooks/87_speckle_reduction>` | Speckle, when the source is a laser. |
| 9 | {doc}`89 Freeform beam shaping <notebooks/89_freeform_beam_shaping>` | Flat-top shapers, optimal transport and freeform lenses. |

**Side trips:** {doc}`10 Interfaces and thin films <notebooks/10_interfaces_thin_films>` and {doc}`20 Polarization ray tracing <notebooks/20_polarization_raytrace>` for coatings and Fresnel losses.

## Laser and beam delivery engineer

For readers who build laser heads, fibre couplings or beam shapers.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`06 Gaussian beams <notebooks/06_gaussian_beams>` | Gaussian and higher order beams. |
| 2 | {doc}`07 Beam quality, M² <notebooks/07_beam_quality>` | Beam widths and M² after ISO 11146. |
| 3 | {doc}`08 ABCD matrices <notebooks/08_abcd_matrices>` | Beam propagation and resonator stability. |
| 4 | {doc}`09 Mode analysis <notebooks/09_mode_analysis>` | Mode content and fibre coupling. |
| 5 | {doc}`56 Gaussian beam decomposition <notebooks/56_gaussian_beam_decomposition>` | A spatial filter with Gaussian beam decomposition. |
| 6 | {doc}`14 Coated laser mirror <notebooks/14_coated_laser_mirror>` | A coated 45° fold mirror: retardance and its effect on the focus. |
| 7 | {doc}`30 IFTA beam shaping <notebooks/30_IFTA>` | Designing a phase-only DOE with IFTA. |
| 8 | {doc}`89 Freeform beam shaping <notebooks/89_freeform_beam_shaping>` | Refractive flat-top shapers and freeform lenses. |
| 9 | {doc}`48 Laser head DOE <notebooks/48_laser_head_doe>` | A diffractive 5 × 5 spot splitter in a processing head. |
| 10 | {doc}`68 Bessel and Airy beams <notebooks/68_bessel_airy_self_healing>` | Bessel and Airy beams. |
| 11 | {doc}`69 Optical vortices <notebooks/69_optical_vortices>` | Vortex beams and mode converters. |
| 12 | {doc}`81 Ultrashort pulses <notebooks/81_ultrashort_pulses_lenses>` | Femtosecond pulses through lenses, prisms and gratings. |
| 13 | {doc}`75 Optical tweezers <notebooks/75_optical_tweezers>` | Optical forces in a tight focus. |

**Side trips:** {doc}`02 Vectorial diffraction <notebooks/02_vectorial_diffraction>` (high NA foci), {doc}`03 Tilted planes <notebooks/03_tilted_planes>`, {doc}`04 Chirp-z transform <notebooks/04_chirp_z_transform>`.

## Metrology, microscopy and computational imaging

For readers who measure wavefronts or build imaging systems that rely on
computation. Three shorter paths share this section.

**Measuring wavefronts and surfaces**

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`18 Image processing <notebooks/18_image_processing>` | Camera noise, quantization and centering. |
| 2 | {doc}`22 Phase unwrapping <notebooks/22_phase_unwrapping>` | Phase unwrapping. |
| 3 | {doc}`23 Interferometry <notebooks/23_interferometry>` | Phase shifting and Fourier fringe analysis. |
| 4 | {doc}`24 Radius interferometer <notebooks/24_radius_interferometer>` | Radius of curvature with a Fizeau interferometer. |
| 5 | {doc}`25 Shack-Hartmann vs phase diversity <notebooks/25_shack_hartmann_vs_phase_diversity>` | Shack-Hartmann sensor against phase diversity. |
| 6 | {doc}`49 Eye aberrometer <notebooks/49_eye_aberrometer>` | A Shack-Hartmann aberrometer behind a schematic eye. |
| 7 | {doc}`28 Phase retrieval <notebooks/28_phase_retrieval>` | Fienup's algorithms and phase diversity. |
| 8 | {doc}`29 TIE <notebooks/29_TIE>` | The transport-of-intensity equation. |
| 9 | {doc}`26 Digital holography <notebooks/26_holography>` | Digital holography. |
| 10 | {doc}`27 Holographic microscopy <notebooks/27_holographic_microscopy>` | Holographic microscopy, combining 22, 26 and 29. |
| 11 | {doc}`76 Schlieren and knife-edge <notebooks/76_schlieren_knife_edge>` | The knife-edge test. |
| 12 | {doc}`82 OCT <notebooks/82_optical_coherence_tomography>` | Depth from coherence: OCT. |

**Microscopy:** {doc}`52 Abbe image formation <notebooks/52_abbe_image_formation>`, {doc}`53 Phase contrast and DIC <notebooks/53_phase_contrast_dic>`, {doc}`54 Confocal microscopy <notebooks/54_confocal_microscopy>`, {doc}`85 Köhler and critical illumination <notebooks/85_koehler_critical_illumination>`, {doc}`78 Super-resolution microscopy <notebooks/78_super_resolution_microscopy>`, {doc}`55 Fourier ptychography <notebooks/55_fourier_ptychography>`.

**Scattering, turbulence and lensless imaging:** {doc}`73 Wavefront shaping <notebooks/73_wavefront_shaping>`, {doc}`72 Twinkling stars and AO <notebooks/72_twinkling_stars_adaptive_optics>`, {doc}`79 Lensless imaging <notebooks/79_lensless_coded_aperture_imaging>`.

**Astronomical instruments:** {doc}`96 Coronagraphs <notebooks/96_coronagraphs>` (Lyot, APLC and vortex coronagraphs, pointing and aberration sensitivity, a dark hole with a deformable mirror), {doc}`97 Telescope zoo <notebooks/97_telescope_zoo>` (Newtonian, Cassegrain, Ritchey–Chrétien and Schmidt camera against the Seidel and Schwarzschild formulas), {doc}`98 Segmented mirrors <notebooks/98_segmented_mirrors>` (hexagon diffraction, piston and tip-tilt errors, dispersed fringe sensing and phase retrieval), {doc}`99 Einstein rings <notebooks/99_einstein_rings>` (gravitational lensing as a GRIN medium and as a thin lens: rings, microlensing, time delays and caustics).

## Nanophotonics and computational electromagnetics

For readers who want to know which solver to use when, and where the
cheaper ones stop being valid.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`00 Foundations and conventions <notebooks/00_foundations_conventions>` | The ladder of approximations and the modules for each rung. |
| 2 | {doc}`01 Free-space propagation <notebooks/01_propagation>` | FFT propagation in homogeneous media. |
| 3 | {doc}`05 Wave propagation method <notebooks/05_WPM>` | The wave propagation method in inhomogeneous media. |
| 4 | {doc}`13 Stacks as transfer functions <notebooks/13_stack_transfer_functions>` | Layer stacks as transfer functions for beams. |
| 5 | {doc}`57 FDTD basics <notebooks/57_fdtd_basics>` | The Yee grid, absorbing boundaries, first FDTD runs. |
| 6 | {doc}`59 FDTD metals and plasmons <notebooks/59_fdtd_metals_plasmons>` | Dispersive metals and surface plasmons. |
| 7 | {doc}`60 FDTD photonic circuits <notebooks/60_fdtd_photonic_circuits>` | Waveguides, couplers and ring resonators. |
| 8 | {doc}`62 FDTD periodic structures <notebooks/62_fdtd_periodic_structures>` | Bloch boundaries, gratings and band structures. |
| 9 | {doc}`63 FDTD devices <notebooks/63_fdtd_devices>` | Photonic crystal devices, a grating coupler and a metalens. |
| 10 | {doc}`64 3D FDTD and nanojet <notebooks/64_fdtd_3d_nanojet>` | 3D FDTD against Mie theory. |
| 11 | {doc}`66 FDFD basics <notebooks/66_fdfd_basics>` | FDFD: one frequency at a time. |
| 12 | {doc}`67 Limaçon microcavity <notebooks/67_fdfd_limacon_microcavity>` | A deformed microcavity with FDFD. |
| 13 | {doc}`80 Carpet cloak <notebooks/80_carpet_cloak>` | Transformation optics. |

**Cross-checks between methods:** {doc}`36 Rigorous mask (RCWA) <notebooks/36_litho_rigorous_mask>`, {doc}`37 Mask models compared <notebooks/37_litho_mask_models>` and {doc}`65 FDTD mask for lithography <notebooks/65_litho_fdtd_mask>` (RCWA against
FDTD), {doc}`77 GRIN optics <notebooks/77_grin_optics>` (rays against FDFD in graded index media).

## Developer extending pyoptics

For readers who want to add a module or a notebook. These notebooks show
how the existing code is organised and which conventions new code has to
keep.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`00 Foundations and conventions <notebooks/00_foundations_conventions>` | The conventions every module follows, checked numerically. Read with {doc}`conventions`. |
| 2 | {doc}`01 Free-space propagation <notebooks/01_propagation>` | Grids, sampling and the basic propagators. |
| 3 | {doc}`04 Chirp-z transform <notebooks/04_chirp_z_transform>` | Zoomed and shifted transforms. |
| 4 | {doc}`46 Analytic phase propagation <notebooks/46_analytic_phase_propagation>` | Splitting off an analytic phase to save samples. |
| 5 | {doc}`16 Basis functions <notebooks/16_bases>` | The common interface of the basis sets. |
| 6 | {doc}`19 Ray tracing to OPD <notebooks/19_ray_tracing>` | The data model of the ray tracer. |
| 7 | {doc}`32 Lithography toolbox <notebooks/32_lithography_toolbox>` | A simulation built from exchangeable components. |
| 8 | {doc}`34 Hopkins imaging and SOCS <notebooks/34_hopkins_socs>` | Abbe against Hopkins: caching what does not change. |
| 9 | {doc}`57 FDTD basics <notebooks/57_fdtd_basics>` | The layout of a time domain solver. |
| 10 | {doc}`66 FDFD basics <notebooks/66_fdfd_basics>` | The same grid as a sparse linear system. |
| 11 | {doc}`18 Image processing <notebooks/18_image_processing>` | Turning simulated fields into realistic measurements. |

See also {doc}`modules` for the maturity of each module.
