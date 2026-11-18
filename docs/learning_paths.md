# Learning paths

The {doc}`examples` are numbered by topic and in the order they were
written, not in the order in which they are best read. This page suggests
paths through them for different readers. Each path has a **core** to read
in order, followed by optional side trips. The notebooks stand on their own,
so leaving a path or mixing two is fine; where a notebook builds on an
earlier one, its introduction usually says so.

Most readers profit from {doc}`conventions` and {doc}`00 Foundations and conventions <notebooks/00_foundations_conventions>` early: they fix the
signs of phases, radii, angles and Zernikes that every other notebook uses.
Readers new to optics can leave notebook 00 (foundations and conventions) for later and read it as a
summary, and start instead with {doc}`103 Introduction to geometrical optics <notebooks/103_geometrical_optics_intro>`:
the lens equation, real and virtual images, principal planes, the
magnifier, microscope and telescope, the Seidel aberrations and the
wavefront error, each formula checked with the ray tracer.

A few notebooks serve almost every path:

- {doc}`52 Abbe image formation <notebooks/52_abbe_image_formation>` is the entry to all of imaging, despite its high number.
- {doc}`17 PSF, MTF, Strehl <notebooks/17_psf_mtf_strehl>` defines the image quality measures that the later notebooks report.
- {doc}`47 Rays and waves <notebooks/47_rays_and_waves>`, {doc}`21 Lens focus from rays to waves <notebooks/21_lens_focus_waves>` and {doc}`56 Gaussian beam decomposition <notebooks/56_gaussian_beam_decomposition>` connect ray tracing with wave optics.

## Undergraduate in a first optics course

For readers who know the textbook theory (Hecht level) and want to see it
work. The core follows the order of a typical lecture.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`103 Introduction to geometrical optics <notebooks/103_geometrical_optics_intro>` | Lens equation, real and virtual images, principal planes, magnifier, microscope and telescope, Seidel aberrations and wavefront error: the ray optics of the first weeks of the course. |
| 2 | {doc}`01 Free-space propagation <notebooks/01_propagation>` | Slits, apertures, Talbot effect and the Poisson spot: diffraction you can check by hand. |
| 3 | {doc}`06 Gaussian beams <notebooks/06_gaussian_beams>` | Gaussian beams as the paraxial building block, and the Gouy phase. |
| 4 | {doc}`08 ABCD matrices <notebooks/08_abcd_matrices>` | Paraxial rays and beams through lenses and resonators with 2×2 matrices. |
| 5 | {doc}`10 Interfaces and thin films <notebooks/10_interfaces_thin_films>` | Fresnel coefficients, Brewster angle, total internal reflection, soap films. |
| 6 | {doc}`11 Jones and Mueller calculus <notebooks/11_polarization>` | Polarization with Jones vectors and Stokes parameters. |
| 7 | {doc}`52 Abbe image formation <notebooks/52_abbe_image_formation>` | Why a microscope has a resolution limit: diffraction orders and the pupil. |
| 8 | {doc}`15 Imaging and Zernikes <notebooks/15_imaging_zernikes>` | Resolution and aberrations described by Zernike polynomials. |
| 9 | {doc}`17 PSF, MTF, Strehl <notebooks/17_psf_mtf_strehl>` | PSF, MTF and Strehl ratio, the language of image quality. |
| 10 | {doc}`23 Interferometry <notebooks/23_interferometry>` | How interferometers turn fringes into wavefronts. |
| 11 | {doc}`74 Coherence and stellar interferometry <notebooks/74_coherence_stellar_interferometry>` | Spatial and temporal coherence, and how they measure stars. |

**Side trips:** {doc}`71 Rainbows and glories <notebooks/71_rainbows_and_glories>` (Descartes' ray picture and Airy's wave picture of
the same bow), {doc}`47 Rays and waves <notebooks/47_rays_and_waves>` (a lab rule explained with rays and waves), {doc}`68 Bessel and Airy beams <notebooks/68_bessel_airy_self_healing>`
(beams that seem not to diffract), {doc}`105 Radiometry and photometry <notebooks/105_radiometry_photometry>` (watts, lumens and why a wall looks equally bright from any distance), {doc}`110 Grating and prism spectrometers <notebooks/110_grating_prism_spectrometers>` sections 1 and 3 (prism against grating, the slit and the resolving power) and {doc}`113 Fabry–Perot and filters <notebooks/113_fabry_perot_filters>` section 1 (the Fabry–Perot as the multiple-beam sequel to two-beam interference),
{doc}`115 Light in a fibre <notebooks/115_fibre_rays_step_graded>` (sections 1 to 4: total internal reflection, the NA and why pulses spread in a
multimode fibre) and {doc}`120 A fibre link <notebooks/120_fibre_link>` (section 1: why telecom uses 1550 nm). Afterwards, {doc}`00 Foundations and conventions <notebooks/00_foundations_conventions>` shows where everything
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
| {doc}`106 Radiometry of images <notebooks/106_radiometry_of_images>` | Why a telescope never shows the Moon brighter than the naked eye, why the corners of a photo are darker, and where the sunny 16 rule comes from. |
| {doc}`102 Sine and Herschel conditions <notebooks/102_sine_herschel_lagrange>` | Why a paraboloid has comet-shaped stars, why a microscope sees one plane sharply, and how remote focusing gets around it. |
| {doc}`80 Carpet cloak <notebooks/80_carpet_cloak>` | Hiding a bump under a carpet with a graded material. |
| {doc}`90 Bokeh <notebooks/90_bokeh>` | Why the background of a fast lens shows cat's eyes, soap bubbles and onion rings. |
| {doc}`91 Sunstars <notebooks/91_sunstars>` | Why a stopped down lens turns the Sun into a star, and how many rays it gets. |
| {doc}`92 Flare and ghosts <notebooks/92_flare_ghosts>` | Where the coloured polygons come from when the Sun is in the frame, and what lens coatings do about them. |
| {doc}`93 The sensor <notebooks/93_sensor>` | Where moiré and false colour come from, what the low pass filter in front of a sensor does, and why some lenses darken the corners on digital cameras. |
| {doc}`94 Depth of field and tilt-shift <notebooks/94_tilt_shift_depth_of_field>` | How much is sharp, when stopping down stops helping, and how a tilted lens puts a whole landscape in focus at f/2.8. |
| {doc}`95 The light field camera <notebooks/95_light_field_camera>` | How a camera with microlenses in front of the sensor can be focused after the picture was taken, and what it pays for that. |
| {doc}`96 Coronagraphs <notebooks/96_coronagraphs>` | How a telescope can see a planet ten billion times fainter than the star next to it. |
| {doc}`101 Atmospheric dispersion corrector <notebooks/101_atmospheric_dispersion_corrector>` | Why stars near the horizon look like little rainbows, and how two rotating prisms in a telescope undo it. |
| {doc}`113 Fabry–Perot and filters <notebooks/113_fabry_perot_filters>` | How the Sun is filmed in the light of a single spectral line with a stack of birefringent plates (section 4). |
| {doc}`114 Spectral lines and absorption <notebooks/114_spectral_lines_absorption>` | Why the CO₂ in a metre of air is already black at 4.3 µm, and why doubling it does not double its absorption (sections 3 and 4). |
| {doc}`100 Echelle spectrograph <notebooks/100_echelle_spectrograph>` | How a spectrograph fits the whole visible spectrum at a resolution of 100 000 onto one detector, and why big telescopes need big gratings. |
| {doc}`99 Einstein rings <notebooks/99_einstein_rings>` | How a galaxy bends light like a lens, turning a distant quasar into a ring or a cross, and why a star passing behind another brightens. |
| {doc}`98 Segmented mirrors <notebooks/98_segmented_mirrors>` | Where the six plus two spikes of JWST images come from, and how 18 mirror segments are aligned to a few nanometres. |
| {doc}`97 Telescope zoo <notebooks/97_telescope_zoo>` | Why big telescopes are Ritchey–Chrétiens, why a Newtonian has comet-shaped stars at the edge, and how the Schmidt camera sees a wide field with a spherical mirror. |

## Lens designer

For readers who use a commercial lens design program and want to compare
pyoptics with it, or go from the ray trace to the wave optical image.
Readers who want a refresher of first order optics and the Seidel
aberrations first can start with {doc}`103 Introduction to geometrical optics <notebooks/103_geometrical_optics_intro>`.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`00 Foundations and conventions <notebooks/00_foundations_conventions>` | Sign conventions for radii, angles, OPD and Zernikes. |
| 2 | {doc}`08 ABCD matrices <notebooks/08_abcd_matrices>` | Paraxial layout with ABCD matrices. |
| 3 | {doc}`43 Delano diagrams <notebooks/43_litho_delano_lens_design>` | First order layout with $y$-$\bar y$ diagrams. |
| 4 | {doc}`19 Ray tracing to OPD <notebooks/19_ray_tracing>` | Exact ray tracing, OPD maps, Zernikes over the field. |
| 5 | {doc}`61 Glass catalogues <notebooks/61_glass_catalogues>` | Glass maps, partial dispersion and an apochromat. |
| 6 | {doc}`44 Seidel aberrations <notebooks/44_seidel_caldwell_lens>` | Seidel sums surface by surface: where correction and tension come from. |
| 7 | {doc}`102 Sine and Herschel conditions <notebooks/102_sine_herschel_lagrange>` | The Lagrange invariant, Conrady's OSC, aplanatic points and why the sine and Herschel conditions exclude each other. |
| 8 | {doc}`58 Lens optimization <notebooks/58_lens_optimization>` | Damped least squares optimization, from a singlet to a microscope objective. |
| 9 | {doc}`45 Zernike aberration gallery <notebooks/45_zernike_aberration_gallery>` | What each Zernike term looks like in a real lens. |
| 10 | {doc}`17 PSF, MTF, Strehl <notebooks/17_psf_mtf_strehl>` | PSF, MTF and Strehl from the pupil. |
| 11 | {doc}`50 Magnification, distortion, telecentricity <notebooks/50_magnification_distortion_telecentricity>` | Magnification, distortion and telecentricity as Zernikes. |
| 12 | {doc}`70 Monte Carlo tolerancing <notebooks/70_monte_carlo_tolerancing>` | Monte Carlo tolerancing. |
| 13 | {doc}`150 Lens tolerances and wafer yield <notebooks/150_lens_tolerances_wafer_yield>` | Tilt, wedge, irregularity and inhomogeneity tolerances, manipulators with limited strokes, inverse sensitivity and tolerance allocation, on a lithography lens. |

**Lens design classics:** {doc}`127 Zoom lenses <notebooks/127_zoom_lenses>` (mechanical compensation, a patent zoom with real glasses, cams and their tolerances, inner focusing and focus breathing), {doc}`128 Forbes Q aspheres <notebooks/128_forbes_q_aspheres>` (slope-orthogonal asphere descriptions, manufacturability and better conditioned optimization), {doc}`159 Phone camera lens with Forbes Q polynomials <notebooks/159_phone_camera_lens_forbes_q>` (a five element f/1.8 plastic phone lens designed with Q-bfs surfaces by widening the field step by step, manufacturing limits on the Q terms, Zernike wavefronts at three wavelengths and Q2D freeform terms), {doc}`129 Progressive spectacle lenses <notebooks/129_progressive_lenses>` (the Minkwitz theorem, hard and soft designs, the lens traced as worn and seen by the eye of {doc}`49 Eye aberrometer <notebooks/49_eye_aberrometer>`), {doc}`149 Scan lenses <notebooks/149_scan_lenses>` (an F-theta and a telecentric scan lens with damped least squares), {doc}`151 Hybrid diffractive lenses <notebooks/151_hybrid_diffractive_lenses>` (kinoforms in the ray tracer, a hybrid achromat and apochromat and a thermal hybrid singlet), {doc}`153 Multi-configuration optimization <notebooks/153_multiconfig_optimization>` (one prescription for several zoom positions or object distances, after {doc}`58 Lens optimization <notebooks/58_lens_optimization>` and {doc}`127 Zoom lenses <notebooks/127_zoom_lenses>`) and {doc}`154 Global optimization and saddle points <notebooks/154_global_optimization_saddle_points>` (the many local minima of a Cooke triplet, the saddle points between them, Bociort's saddle-point construction from doublets to triplets, basin hopping against random starts), and {doc}`155 Off-axis mirrors and nodal aberrations <notebooks/155_off_axis_mirrors_nodal_aberrations>` (off-axis segments, nodal aberration theory of a misaligned Ritchey–Chrétien with constant coma and binodal astigmatism, an unobscured three-mirror anastigmat with ring fields and freeform mirrors, after {doc}`97 Telescope zoo <notebooks/97_telescope_zoo>`).

**From rays to waves:** {doc}`21 Lens focus from rays to waves <notebooks/21_lens_focus_waves>`, {doc}`47 Rays and waves <notebooks/47_rays_and_waves>`, {doc}`46 Analytic phase propagation <notebooks/46_analytic_phase_propagation>` (propagating strongly curved
wavefronts), {doc}`56 Gaussian beam decomposition <notebooks/56_gaussian_beam_decomposition>`.
**Polarization and coatings:** {doc}`20 Polarization ray tracing <notebooks/20_polarization_raytrace>`, {doc}`14 Coated laser mirror <notebooks/14_coated_laser_mirror>`.
**Broadband light:** {doc}`81 Ultrashort pulses <notebooks/81_ultrashort_pulses_lenses>`.
**Camera lenses:** {doc}`90 Bokeh <notebooks/90_bokeh>` (a double Gauss, vignetting, out-of-focus rendering), {doc}`91 Sunstars <notebooks/91_sunstars>` (diffraction at the iris), {doc}`92 Flare and ghosts <notebooks/92_flare_ghosts>` (double reflections, coatings), {doc}`93 The sensor <notebooks/93_sensor>` (pixels, Bayer mosaic, low pass filter, microlenses), {doc}`94 Depth of field and tilt-shift <notebooks/94_tilt_shift_depth_of_field>` (hyperfocal distance, optimum aperture, Scheimpflug), {doc}`95 The light field camera <notebooks/95_light_field_camera>` (microlenses, views, refocusing), {doc}`106 Radiometry of images <notebooks/106_radiometry_of_images>` (the camera equation, cos⁴ and how a retrofocus lens beats it, exposure and photons per pixel). {doc}`160 Image restoration and PSF deconvolution <notebooks/160_image_restoration_deconvolution>` (field and colour dependent PSFs of a phone lens, Wiener, Richardson–Lucy, TV and cross-channel deconvolution, the Bayer mosaic, wrong PSFs and a learned unrolled solver), {doc}`161 Distortion, lateral colour and lens shading correction <notebooks/161_distortion_shading_correction>` (the distortion of the phone lens and its warp, a warp per colour channel, warp before or after the restoration, relative illumination, chief ray angle and microlens shift, colour shading from the IR cut filter, ISP gain tables).

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
(overlay), {doc}`43 Delano diagrams <notebooks/43_litho_delano_lens_design>`, {doc}`44 Seidel aberrations <notebooks/44_seidel_caldwell_lens>`, {doc}`70 Monte Carlo tolerancing <notebooks/70_monte_carlo_tolerancing>`, {doc}`150 Lens tolerances and wafer yield <notebooks/150_lens_tolerances_wafer_yield>` (from lens tolerances to CD uniformity, best focus and overlay, and what the manipulators recover), {doc}`42 KrF illuminator <notebooks/42_litho_illuminator>`, {doc}`86 Homogenizers <notebooks/86_homogenizers>`.
**Computational lithography:** {doc}`39 Source optimization <notebooks/39_litho_source_optimization>`, then {doc}`40 OPC <notebooks/40_litho_opc>`, then {doc}`108 Arbitrary mask layouts <notebooks/108_litho_arbitrary_layouts>` (polygons at any angle, layouts from gdstk/gdspy).

**Beyond the mean image:** {doc}`152 Litho stochastics, flare and focus blur <notebooks/152_litho_stochastics_flare_focus_blur>` (photon and acid shot noise, LER, LWR and their spectra, local CDU and missing contacts; flare from lens roughness and pattern density; focus blur from the laser bandwidth and the stage), after {doc}`35 Resist and swing curves <notebooks/35_litho_resist_swing>`.

**EUV:** {doc}`156 EUV multilayers and reflective masks <notebooks/156_euv_multilayers_reflective_masks>` (Mo/Si Bragg mirrors with Ru cap, their Jones pupil; a TaBN or low-n absorber at a 6° chief ray: shadowing, best focus through pitch, pattern shift, telecentricity) → {doc}`157 EUV projector and pupil <notebooks/157_euv_projector_pupil>` (NA 0.33 against anamorphic NA 0.55, central obscuration and forbidden pitches, a Schwarzschild mirror objective with graded coatings) → {doc}`158 EUV stochastics and flare <notebooks/158_euv_stochastics_flare>` (13.5 nm photon shot noise, secondary electron blur, Z factor, contact LCDU and flare scaling as 1/λ²), after {doc}`152 Litho stochastics, flare and focus blur <notebooks/152_litho_stochastics_flare_focus_blur>`.
**Wafer metrology:** {doc}`135 Ellipsometry <notebooks/135_ellipsometry>` (film thicknesses and optical constants), then {doc}`136 Scatterometry <notebooks/136_scatterometry>` (CD, height and sidewall angle of printed lines with RCWA).

## Illumination engineer

For readers who think in étendue, irradiance and uniformity rather than in
PSFs. This group is nearly self-contained; {doc}`19 Ray tracing to OPD <notebooks/19_ray_tracing>` introduces the ray tracer
that it uses.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`105 Radiometry and photometry <notebooks/105_radiometry_photometry>` | Flux, intensity, irradiance and radiance; lumens and luminous efficacy; view factors and the integrating sphere. |
| 2 | {doc}`83 Étendue <notebooks/83_etendue_brightness_theorem>` | Étendue, phase space and the brightness theorem. |
| 3 | {doc}`84 Nonimaging concentrators <notebooks/84_nonimaging_concentrators>` | The compound parabolic concentrator and the edge ray principle. |
| 4 | {doc}`88 LED optics <notebooks/88_led_optics_monte_carlo>` | LED optics and Monte Carlo illuminance. |
| 5 | {doc}`86 Homogenizers <notebooks/86_homogenizers>` | Fly's eye lenslet arrays and integrator rods. |
| 6 | {doc}`85 Köhler and critical illumination <notebooks/85_koehler_critical_illumination>` | Köhler and critical illumination. |
| 7 | {doc}`51 Fresnel lenses <notebooks/51_fresnel_lenses>` | Fresnel lenses and solar concentrators. |
| 8 | {doc}`42 KrF illuminator <notebooks/42_litho_illuminator>` | A complete illuminator, ray traced. |
| 9 | {doc}`87 Speckle reduction <notebooks/87_speckle_reduction>` | Speckle, when the source is a laser. |
| 10 | {doc}`89 Freeform beam shaping <notebooks/89_freeform_beam_shaping>` | Flat-top shapers, optimal transport and freeform lenses. |

**Side trips:** {doc}`10 Interfaces and thin films <notebooks/10_interfaces_thin_films>` and {doc}`20 Polarization ray tracing <notebooks/20_polarization_raytrace>` for coatings and Fresnel losses; {doc}`106 Radiometry of images <notebooks/106_radiometry_of_images>` for how bright an image is (the camera equation and the cos⁴ law).

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
| 10 | {doc}`149 Scan lenses <notebooks/149_scan_lenses>` | F-theta and telecentric scan lenses behind two galvos, the bow of the scan lines and its lookup table. |
| 11 | {doc}`68 Bessel and Airy beams <notebooks/68_bessel_airy_self_healing>` | Bessel and Airy beams. |
| 12 | {doc}`69 Optical vortices <notebooks/69_optical_vortices>` | Vortex beams and mode converters. |
| 13 | {doc}`81 Ultrashort pulses <notebooks/81_ultrashort_pulses_lenses>` | Femtosecond pulses through lenses, prisms and gratings. |
| 14 | {doc}`75 Optical tweezers <notebooks/75_optical_tweezers>` | Optical forces in a tight focus. |

**Side trips:** {doc}`02 Vectorial diffraction <notebooks/02_vectorial_diffraction>` (high NA foci), {doc}`03 Tilted planes <notebooks/03_tilted_planes>`, {doc}`04 Chirp-z transform <notebooks/04_chirp_z_transform>`; for fibre delivery,
{doc}`115 Light in a fibre <notebooks/115_fibre_rays_step_graded>` (NA, étendue and bend loss of multimode fibres), {doc}`116 Fibre modes <notebooks/116_fibre_modes>`
(single-mode condition and mode field diameter) and {doc}`119 Fibre Bragg gratings <notebooks/119_fibre_bragg_gratings>` (gratings that stabilize lasers).

## Optical communication and fibre optics

For readers who want to understand how light travels in fibres and what
limits a fibre link, at the level of a first course on optical
communication. Real data (GeO₂-doped silica after Fleming, a G.652
standard fibre) and practical units (µm, km, ps, ps/(nm km), dBm)
throughout.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`09 Mode analysis <notebooks/09_mode_analysis>` | LP modes of a step-index fibre and coupling a Gaussian beam into it. |
| 2 | {doc}`115 Light in a fibre <notebooks/115_fibre_rays_step_graded>` | NA and acceptance angle, skew rays, graded-index rays and modal dispersion in the ray picture. |
| 3 | {doc}`116 Fibre modes <notebooks/116_fibre_modes>` | From rays to modes, the single-mode condition, mode field diameter, splice and bend loss. |
| 4 | {doc}`117 Dispersion in fibres <notebooks/117_fibre_dispersion>` | Material, waveguide and intermodal dispersion, the zero-dispersion wavelength, PMD. |
| 5 | {doc}`118 Pulse broadening <notebooks/118_fibre_pulse_broadening>` | Pulses in dispersive fibre, the source width and the bit-rate limit of each fibre type. |
| 6 | {doc}`119 Fibre Bragg gratings <notebooks/119_fibre_bragg_gratings>` | Coupled modes, apodization, chirped gratings that undo dispersion, drop filters and sensors. |
| 7 | {doc}`120 A fibre link <notebooks/120_fibre_link>` | Loss, receivers and noise, eye diagrams, the bit error rate and link budgets. |
| 8 | {doc}`121 SPM and solitons in fibres <notebooks/121_fibre_spm_solitons>` | The Kerr effect in a fibre: self-phase modulation, solitons, supercontinuum, the Raman self-frequency shift and the power limit of a link. |

**Side trips:** {doc}`10 Interfaces and thin films <notebooks/10_interfaces_thin_films>` (total internal reflection and its
phase), {doc}`77 GRIN optics <notebooks/77_grin_optics>` (curved rays in graded media), {doc}`60 FDTD photonic circuits <notebooks/60_fdtd_photonic_circuits>` (waveguides,
couplers and rings on a chip), {doc}`81 Ultrashort pulses <notebooks/81_ultrashort_pulses_lenses>` (group delay dispersion in glass), {doc}`14 Coated laser mirror <notebooks/14_coated_laser_mirror>`
(Bragg mirrors as thin films), {doc}`112 Field tracing applications <notebooks/112_field_tracing_applications>` (coupling a beam into a single-mode
fibre with a lens).

## AR/VR and display optics

For readers who work on head-mounted displays or head-up displays and
want to see where the numbers of a datasheet (field of view, eye box,
ghosts, image distance) come from. Real glasses from the SCHOTT
catalogue, crystal quartz and windshield geometry throughout.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`10 Interfaces and thin films <notebooks/10_interfaces_thin_films>` | Total internal reflection, Brewster's angle and Fresnel reflectance, on which everything below rests. |
| 2 | {doc}`123 AR waveguide combiners <notebooks/123_ar_waveguide_combiners>` | The field of view in k-space, slanted in-couplers with RCWA, pupil replication, the eye box and the rainbow artefact. |
| 3 | {doc}`124 Volume holographic gratings <notebooks/124_volume_holographic_gratings>` | Kogelnik's coupled waves, Bragg selectivity and why holographic combiners trade field of view for transparency. |
| 4 | {doc}`125 Pancake lenses <notebooks/125_pancake_lenses>` | Folding a VR eyepiece with polarization, and the ghosts of real retarders over the spectrum and the field. |
| 5 | {doc}`126 Head-up displays <notebooks/126_head_up_displays>` | The windshield as an off-axis mirror: double images, wedges and a freeform HUD mirror judged by vergence and disparity. |

**Side trips:** {doc}`11 Jones and Mueller calculus <notebooks/11_polarization>` (the polarization
algebra behind the pancake), {doc}`36 Rigorous mask (RCWA) <notebooks/36_litho_rigorous_mask>` (the method behind the grating
efficiencies), {doc}`110 Grating and prism spectrometers <notebooks/110_grating_prism_spectrometers>` (off-axis mirrors and
Coddington astigmatism in a Czerny–Turner), {doc}`51 Fresnel lenses <notebooks/51_fresnel_lenses>` (the VR lens before
pancakes), {doc}`83 Étendue <notebooks/83_etendue_brightness_theorem>` (why a small projector cannot fill a large eye box
for free).

## 3D sensing and LiDAR

For readers who build or choose depth sensors and want to know where
their range, precision and eye safety limits come from: photon budgets,
detectors, scanners and triangulation geometry, with the numbers of
automotive LiDARs and phone depth cameras.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`105 Radiometry and photometry <notebooks/105_radiometry_photometry>` | Radiance, solid angles and Lambertian surfaces, the language of every link budget. |
| 2 | {doc}`130 Time-of-flight LiDAR <notebooks/130_time_of_flight_lidar>` | The LiDAR equation, sunlight, APDs and SPADs with pile-up, and the class 1 limits at 905 and 1550 nm. |
| 3 | {doc}`131 FMCW LiDAR <notebooks/131_fmcw_lidar>` | Chirps, range and Doppler, the shot noise limited coherent receiver, speckle fading and the laser it needs. |
| 4 | {doc}`132 Beam scanning <notebooks/132_beam_scanning>` | Galvos, polygons, MEMS mirrors, phased arrays and Risley prisms, and the invariant of resolvable spots. |
| 5 | {doc}`133 Triangulation and structured light <notebooks/133_triangulation_structured_light>` | Laser triangulation and its speckle limit, fringe projection with temporal unwrapping, and the dot projector of depth cameras. |

**Side trips:** {doc}`120 Fibre link <notebooks/120_fibre_link>` (the APD noise model and
Q factors), {doc}`83 Étendue <notebooks/83_etendue_brightness_theorem>` (why a single mode receiver
sees no sunlight), {doc}`87 Speckle reduction <notebooks/87_speckle_reduction>` (speckle statistics
and diversity), {doc}`23 Interferometry <notebooks/23_interferometry>` and {doc}`22 Phase unwrapping <notebooks/22_phase_unwrapping>`
(phase shifting and unwrapping behind fringe projection), {doc}`30 IFTA beam shaping <notebooks/30_IFTA>`
(the algorithm behind the fan-out grating).

## Surface and thin-film metrology

For readers who measure surfaces, films and patterned wafers, or choose
the instruments that do: what an interferometer, an ellipsometer or a
deflectometer actually measures, how the numbers come out of models and
fits, and where the systematic errors hide.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`10 Interfaces and thin films <notebooks/10_interfaces_thin_films>` | Fresnel coefficients and the characteristic matrices of thin film stacks, the forward model of everything below. |
| 2 | {doc}`135 Ellipsometry <notebooks/135_ellipsometry>` | Psi and Delta, rotating analyzers and compensators, dispersion models and fits, and the correlation of thickness and index. |
| 3 | {doc}`136 Scatterometry <notebooks/136_scatterometry>` | Line profiles from the zero order of a grating with RCWA: sensitivities, libraries, regression and model errors. |
| 4 | {doc}`82 OCT <notebooks/82_optical_coherence_tomography>` | Depth from the coherence of broadband light, the physics of white-light interferometry. |
| 5 | {doc}`134 Coherence scanning interferometry (white light interferometry, WLI) <notebooks/134_coherence_scanning_interferometry>` | Surface heights from envelope and phase, NA, metals, films and batwings. |
| 6 | {doc}`133 Triangulation and structured light <notebooks/133_triangulation_structured_light>` | Phase shifted fringes and temporal unwrapping, used again by deflectometry. |
| 7 | {doc}`137 Deflectometry <notebooks/137_deflectometry>` | Slopes of specular surfaces from a screen and a camera, the height-slope ambiguity and calibration. |

**Side trips:** {doc}`23 Interferometry <notebooks/23_interferometry>` (phase shifting and Fourier
fringe analysis), {doc}`24 Radius interferometer <notebooks/24_radius_interferometer>` (a Fizeau
interferometer, the low orders deflectometry is weak on), {doc}`36 Rigorous mask (RCWA) <notebooks/36_litho_rigorous_mask>`
(the RCWA behind scatterometry), {doc}`128 Forbes Q aspheres <notebooks/128_forbes_q_aspheres>` (the
slope orthogonal polynomials that describe what deflectometry measures).

## Crystal optics and light modulators

For readers who use polarizers, wave plates, isolators and modulators, or
design with them: where birefringence comes from, how a voltage or a
sound wave changes it, and what the devices built on it can and cannot do.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`11 Jones and Mueller calculus <notebooks/11_polarization>` | Jones vectors and matrices, retarders and polarizers, the language of everything below. |
| 2 | {doc}`138 Crystal optics <notebooks/138_crystal_optics>` | The index ellipsoid, walk-off, the 4×4 matrix method, polarizing prisms, optical activity and the Faraday isolator. |
| 3 | {doc}`139 Electro-optic modulators <notebooks/139_electro_optic_modulators>` | Pockels and Kerr cells, the Mach–Zehnder modulator, its chirp, extinction and bandwidth. |
| 4 | {doc}`140 Liquid crystals <notebooks/140_liquid_crystals>` | The Fréedericksz transition, director profiles, LC retarders and the TN cell with its viewing angle. |
| 5 | {doc}`141 Acousto-optics <notebooks/141_acousto_optics>` | Raman–Nath and Bragg diffraction, AOMs, AODs and the acousto-optic tunable filter. |
| 6 | {doc}`113 Fabry–Perot and filters <notebooks/113_fabry_perot_filters>` | Lyot and Šolc filters from real calcite plates, their dispersion and field of view. |

**Side trips:** {doc}`120 Fibre link <notebooks/120_fibre_link>` (the modulator in a
10 Gb/s link: chirp against dispersion), {doc}`122 Imaging spectroscopy <notebooks/122_imaging_spectroscopy>`
(an LC tunable filter and an AOTF in a hyperspectral camera), {doc}`125 Pancake lenses <notebooks/125_pancake_lenses>`
(real retarders in a VR lens), {doc}`20 Polarization ray tracing <notebooks/20_polarization_raytrace>`
(crystals in a sequential ray trace).

## Lasers

For readers who use lasers or build them: where the gain comes from, what
sets the threshold, the output and the linewidth, how pulses are made, and
how a laser diode behaves when its current is modulated.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`08 ABCD matrices <notebooks/08_abcd_matrices>` | Stable resonators, their Gaussian modes and Gouy phases. |
| 2 | {doc}`142 Laser fundamentals <notebooks/142_laser_fundamentals>` | Cross sections, saturation, threshold, slope efficiency, output coupling, modes, linewidth and relaxation oscillations. |
| 3 | {doc}`143 Q-switching and mode-locking <notebooks/143_q_switching_mode_locking>` | Giant pulses from Q-switching, locked modes and combs, active and soliton mode-locking. |
| 4 | {doc}`144 Laser diodes <notebooks/144_laser_diodes>` | The L–I curve, modulation bandwidth, chirp, RIN and linewidth of a DFB laser. |
| 5 | {doc}`120 Fibre link <notebooks/120_fibre_link>` | The DFB laser as the transmitter of a 10 Gb/s link: its chirp against dispersion and its RIN. |

**Side trips:** {doc}`81 Ultrashort pulses <notebooks/81_ultrashort_pulses_lenses>` (what
happens to a mode-locked pulse in a lens), {doc}`114 Spectral lines and absorption <notebooks/114_spectral_lines_absorption>`
(line shapes behind the cross sections), {doc}`113 Fabry–Perot and filters <notebooks/113_fabry_perot_filters>`
(the resonator as a filter) and {doc}`147 Kerr effect and self-focusing <notebooks/147_kerr_effect_self_focusing>`
(section 6: Kerr-lens mode-locking from a cavity design).

## Nonlinear optics

For readers who double, mix or broaden laser light, or need to know when
their beam starts to do it on its own: the second-order processes that
make new frequencies in crystals, the third-order Kerr effect that
focuses beams and broadens pulses, and the split-step solver behind the
pulse and beam simulations.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`138 Crystal optics <notebooks/138_crystal_optics>` | Ordinary and extraordinary waves, the index ellipsoid and walk-off, which phase matching relies on. |
| 2 | {doc}`145 Second harmonic generation <notebooks/145_second_harmonic_generation>` | χ(2), the coupled amplitude equations, birefringent and quasi phase matching, acceptance bandwidths, Boyd–Kleinman focusing and femtosecond SHG. |
| 3 | {doc}`146 Parametric amplification and OPOs <notebooks/146_parametric_amplification_opo>` | Sum and difference frequencies, Manley–Rowe, parametric gain and bandwidth, tuning curves and OPO thresholds. |
| 4 | {doc}`147 Kerr effect and self-focusing <notebooks/147_kerr_effect_self_focusing>` | n₂, the Kerr lens, critical power and collapse, filamentation seeds, SPM in bulk and Kerr-lens mode-locking. |
| 5 | {doc}`121 SPM and solitons in fibres <notebooks/121_fibre_spm_solitons>` | The nonlinear Schrödinger equation, split-step accuracy, solitons, supercontinuum and the Raman self-frequency shift. |

**Side trips:** {doc}`143 Q-switching and mode-locking <notebooks/143_q_switching_mode_locking>`
(the master equation that the Kerr-lens loss plugs into), {doc}`81 Ultrashort pulses <notebooks/81_ultrashort_pulses_lenses>`
(dispersion of femtosecond pulses in glass) and {doc}`117 Dispersion in fibres <notebooks/117_fibre_dispersion>`
(the β₂ that balances SPM in a soliton).

## Detectors and photon statistics

For readers who choose, use or design detectors, or count photons: where
responsivity, bandwidth and noise come from, how to compare detectors,
and what the statistics of the photons say about the light.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`105 Radiometry and photometry <notebooks/105_radiometry_photometry>` | Watts, photons and black bodies: the units a detector sees. |
| 2 | {doc}`148 Detectors and photon statistics <notebooks/148_detectors_photon_statistics>` | Responsivity, band gaps and absorption, bandwidth, NEP and D*, SPADs, photon statistics, HBT and the HOM dip. |
| 3 | {doc}`120 Fibre link <notebooks/120_fibre_link>` | PIN and APD receivers at work: Q factor, bit error rate and sensitivity. |
| 4 | {doc}`130 Time-of-flight LiDAR <notebooks/130_time_of_flight_lidar>` | Linear APDs against SPADs with pile-up in a LiDAR link budget. |
| 5 | {doc}`18 Image processing <notebooks/18_image_processing>` | Camera noise in images: photon shot noise, read noise and the ADC. |

**Side trips:** {doc}`111 Fourier transform spectroscopy <notebooks/111_fourier_transform_spectroscopy>`
(the multiplex advantage of detector limited instruments), {doc}`74 Coherence and stellar interferometry <notebooks/74_coherence_stellar_interferometry>`
(intensity interferometry after Hanbury Brown and Twiss) and {doc}`146 Parametric amplification and OPOs <notebooks/146_parametric_amplification_opo>`
(the parametric process that makes photon pairs).

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

**Spectroscopy:** {doc}`110 Grating and prism spectrometers <notebooks/110_grating_prism_spectrometers>` (dispersive spectrometers, the traced Czerny–Turner, grating efficiency), then {doc}`111 Fourier transform spectroscopy <notebooks/111_fourier_transform_spectroscopy>` (the FTIR: apodization, sampling, phase correction, Jacquinot and Fellgett), then {doc}`114 Spectral lines and absorption <notebooks/114_spectral_lines_absorption>` (line shapes, the curve of growth, and why concentrations from unresolved lines go wrong); {doc}`113 Fabry–Perot and filters <notebooks/113_fabry_perot_filters>` for etalons and filters and {doc}`122 Imaging spectroscopy <notebooks/122_imaging_spectroscopy>` for pushbroom, filter and snapshot imaging spectrometers.

**Astronomical instruments:** {doc}`96 Coronagraphs <notebooks/96_coronagraphs>` (Lyot, APLC and vortex coronagraphs, pointing and aberration sensitivity, a dark hole with a deformable mirror), {doc}`97 Telescope zoo <notebooks/97_telescope_zoo>` (Newtonian, Cassegrain, Ritchey–Chrétien and Schmidt camera against the Seidel and Schwarzschild formulas), {doc}`98 Segmented mirrors <notebooks/98_segmented_mirrors>` (hexagon diffraction, piston and tip-tilt errors, dispersed fringe sensing and phase retrieval), {doc}`99 Einstein rings <notebooks/99_einstein_rings>` (gravitational lensing as a GRIN medium and as a thin lens: rings, microlensing, time delays and caustics), {doc}`100 Echelle spectrograph <notebooks/100_echelle_spectrograph>` (grating equation in high orders, blaze, free spectral range, cross-dispersion, the echellogram and the resolving power against the slit), {doc}`101 Atmospheric dispersion corrector <notebooks/101_atmospheric_dispersion_corrector>` (refraction and dispersion of air against the zenith angle, counter-rotating Amici prisms, a glass pair matched to air, the residual against the zenith angle).

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
| 13 | {doc}`104 Phase space, Husimi and ray chaos <notebooks/104_phase_space_husimi_chaos>` | Wigner and Husimi functions; the modes of chaotic cavities against billiard rays. |
| 14 | {doc}`80 Carpet cloak <notebooks/80_carpet_cloak>` | Transformation optics. |

**Cross-checks between methods:** {doc}`36 Rigorous mask (RCWA) <notebooks/36_litho_rigorous_mask>`, {doc}`37 Mask models compared <notebooks/37_litho_mask_models>` and {doc}`65 FDTD mask for lithography <notebooks/65_litho_fdtd_mask>` (RCWA against
FDTD), {doc}`77 GRIN optics <notebooks/77_grin_optics>` (rays against FDFD in graded index media), {doc}`110 Grating and prism spectrometers <notebooks/110_grating_prism_spectrometers>` section 4 (RCWA efficiency of a blazed grating, TE and TM, Wood–Rayleigh anomalies).

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
