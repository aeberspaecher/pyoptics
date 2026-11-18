# Examples

The example notebooks illustrate the modules with examples whose physics
is easy to follow, and compare the numerics with analytic results wherever
an analytic solution exists. They are shown here with their stored outputs;
to run them yourself, see [Running the notebooks](#running-the-notebooks).
For suggested reading orders by background and interest, see
{doc}`learning_paths`.

## Foundations and conventions

From Maxwell's equations down to rays, with the modules for each level of approximation and numerical checks of all sign conventions. A first course in geometrical optics (images, principal planes, simple instruments, Seidel aberrations and wavefront error) for readers new to optics.

- {doc}`notebooks/00_foundations_conventions`
- {doc}`notebooks/103_geometrical_optics_intro`

```{toctree}
:hidden:
:caption: Foundations

notebooks/00_foundations_conventions
notebooks/103_geometrical_optics_intro
```

## Propagation and diffraction

Scalar and vectorial propagation of fields between planes, onto tilted planes and through inhomogeneous media.

- {doc}`notebooks/01_propagation`
- {doc}`notebooks/02_vectorial_diffraction`
- {doc}`notebooks/03_tilted_planes`
- {doc}`notebooks/04_chirp_z_transform`
- {doc}`notebooks/05_WPM`
- {doc}`notebooks/46_analytic_phase_propagation`

```{toctree}
:hidden:
:caption: Propagation and diffraction

notebooks/01_propagation
notebooks/02_vectorial_diffraction
notebooks/03_tilted_planes
notebooks/04_chirp_z_transform
notebooks/05_WPM
notebooks/46_analytic_phase_propagation
```

## Maxwell solvers: FDTD and FDFD

Full-wave solutions of Maxwell's equations on the Yee grid, in the time domain (`fdtd`) and one frequency at a time (`fdfd`).

- {doc}`notebooks/57_fdtd_basics`
- {doc}`notebooks/59_fdtd_metals_plasmons`
- {doc}`notebooks/60_fdtd_photonic_circuits`
- {doc}`notebooks/62_fdtd_periodic_structures`
- {doc}`notebooks/63_fdtd_devices`
- {doc}`notebooks/64_fdtd_3d_nanojet`
- {doc}`notebooks/66_fdfd_basics`
- {doc}`notebooks/67_fdfd_limacon_microcavity`
- {doc}`notebooks/104_phase_space_husimi_chaos`
- {doc}`notebooks/80_carpet_cloak`
- {doc}`notebooks/65_litho_fdtd_mask` (in *Lithography*)

```{toctree}
:hidden:
:caption: Maxwell solvers: FDTD and FDFD

notebooks/57_fdtd_basics
notebooks/59_fdtd_metals_plasmons
notebooks/60_fdtd_photonic_circuits
notebooks/62_fdtd_periodic_structures
notebooks/63_fdtd_devices
notebooks/64_fdtd_3d_nanojet
notebooks/66_fdfd_basics
notebooks/67_fdfd_limacon_microcavity
notebooks/104_phase_space_husimi_chaos
notebooks/80_carpet_cloak
```

## Beams, modes and paraxial optics

Gaussian and structured beams, beam quality, resonators, modes and ultrashort pulses.

- {doc}`notebooks/06_gaussian_beams`
- {doc}`notebooks/07_beam_quality`
- {doc}`notebooks/08_abcd_matrices`
- {doc}`notebooks/09_mode_analysis`
- {doc}`notebooks/68_bessel_airy_self_healing`
- {doc}`notebooks/69_optical_vortices`
- {doc}`notebooks/81_ultrashort_pulses_lenses`

```{toctree}
:hidden:
:caption: Beams, modes and paraxial optics

notebooks/06_gaussian_beams
notebooks/07_beam_quality
notebooks/08_abcd_matrices
notebooks/09_mode_analysis
notebooks/68_bessel_airy_self_healing
notebooks/69_optical_vortices
notebooks/81_ultrashort_pulses_lenses
```

## Fibre optics and optical communication

Light guided in glass fibres and the basics of sending bits through them:
the acceptance angle and rays in step and graded-index cores, fibre modes
and the single-mode condition, material, waveguide and intermodal
dispersion with real GeO₂–SiO₂ data, pulse broadening and the bit-rate
limit, fibre Bragg gratings, a basic link with its receiver noise, eye
diagrams and budgets, and self-phase modulation, solitons and
supercontinuum. Related: LP modes and fibre coupling in
{doc}`notebooks/09_mode_analysis`, waveguides in
{doc}`notebooks/60_fdtd_photonic_circuits`, GRIN rods in
{doc}`notebooks/77_grin_optics` and pulses in glass in
{doc}`notebooks/81_ultrashort_pulses_lenses`.

- {doc}`notebooks/115_fibre_rays_step_graded`
- {doc}`notebooks/116_fibre_modes`
- {doc}`notebooks/117_fibre_dispersion`
- {doc}`notebooks/118_fibre_pulse_broadening`
- {doc}`notebooks/119_fibre_bragg_gratings`
- {doc}`notebooks/120_fibre_link`
- {doc}`121 SPM and solitons in fibres <notebooks/121_fibre_spm_solitons>`

```{toctree}
:hidden:
:caption: Fibre optics and optical communication

notebooks/115_fibre_rays_step_graded
notebooks/116_fibre_modes
notebooks/117_fibre_dispersion
notebooks/118_fibre_pulse_broadening
notebooks/119_fibre_bragg_gratings
notebooks/120_fibre_link
notebooks/121_fibre_spm_solitons
```

## Thin films, coatings and polarization

Fresnel coefficients, layer stacks and coatings, Jones and Mueller calculus.

- {doc}`notebooks/10_interfaces_thin_films`
- {doc}`notebooks/11_polarization`
- {doc}`notebooks/12_layer_stacks`
- {doc}`notebooks/13_stack_transfer_functions`
- {doc}`notebooks/14_coated_laser_mirror`

```{toctree}
:hidden:
:caption: Thin films, coatings and polarization

notebooks/10_interfaces_thin_films
notebooks/11_polarization
notebooks/12_layer_stacks
notebooks/13_stack_transfer_functions
notebooks/14_coated_laser_mirror
```

## Imaging, aberrations and detection

Pupils and Zernike polynomials, PSF, MTF and Strehl ratio, detectors and lensless imaging.

- {doc}`notebooks/15_imaging_zernikes`
- {doc}`notebooks/16_bases`
- {doc}`notebooks/17_psf_mtf_strehl`
- {doc}`notebooks/18_image_processing`
- {doc}`notebooks/45_zernike_aberration_gallery`
- {doc}`notebooks/79_lensless_coded_aperture_imaging`

```{toctree}
:hidden:
:caption: Imaging, aberrations and detection

notebooks/15_imaging_zernikes
notebooks/16_bases
notebooks/17_psf_mtf_strehl
notebooks/18_image_processing
notebooks/45_zernike_aberration_gallery
notebooks/79_lensless_coded_aperture_imaging
```

## Ray tracing and lens design

Sequential and polarization ray tracing, from rays to waves, glass catalogues, lens optimization and tolerancing, and what a lens must conserve: the Lagrange invariant, the sine and the Herschel condition, and field tracing: rays through the lenses, waves where diffraction matters. Three lens design classics close the section: a mechanically compensated zoom from a patent, Forbes Q aspheres, and a progressive spectacle lens seen by a model eye.

- {doc}`notebooks/19_ray_tracing`
- {doc}`notebooks/20_polarization_raytrace`
- {doc}`notebooks/21_lens_focus_waves`
- {doc}`notebooks/47_rays_and_waves`
- {doc}`notebooks/48_laser_head_doe`
- {doc}`notebooks/49_eye_aberrometer`
- {doc}`notebooks/51_fresnel_lenses`
- {doc}`notebooks/56_gaussian_beam_decomposition`
- {doc}`notebooks/58_lens_optimization`
- {doc}`notebooks/61_glass_catalogues`
- {doc}`notebooks/70_monte_carlo_tolerancing`
- {doc}`notebooks/77_grin_optics`
- {doc}`notebooks/102_sine_herschel_lagrange`
- {doc}`notebooks/109_field_tracing`
- {doc}`notebooks/112_field_tracing_applications`
- {doc}`notebooks/127_zoom_lenses`
- {doc}`notebooks/128_forbes_q_aspheres`
- {doc}`notebooks/129_progressive_lenses`
- {doc}`notebooks/149_scan_lenses`
- {doc}`151 Hybrid diffractive lenses <notebooks/151_hybrid_diffractive_lenses>`
- {doc}`153 Multi-configuration optimization <notebooks/153_multiconfig_optimization>`
- {doc}`154 Global optimization and saddle points <notebooks/154_global_optimization_saddle_points>`
- {doc}`155 Off-axis mirrors and nodal aberrations <notebooks/155_off_axis_mirrors_nodal_aberrations>`
- {doc}`159 Phone camera lens with Forbes Q polynomials <notebooks/159_phone_camera_lens_forbes_q>`

```{toctree}
:hidden:
:caption: Ray tracing and lens design

notebooks/19_ray_tracing
notebooks/20_polarization_raytrace
notebooks/21_lens_focus_waves
notebooks/47_rays_and_waves
notebooks/48_laser_head_doe
notebooks/49_eye_aberrometer
notebooks/51_fresnel_lenses
notebooks/56_gaussian_beam_decomposition
notebooks/58_lens_optimization
notebooks/61_glass_catalogues
notebooks/70_monte_carlo_tolerancing
notebooks/77_grin_optics
notebooks/102_sine_herschel_lagrange
notebooks/109_field_tracing
notebooks/112_field_tracing_applications
notebooks/127_zoom_lenses
notebooks/128_forbes_q_aspheres
notebooks/129_progressive_lenses
notebooks/149_scan_lenses
notebooks/151_hybrid_diffractive_lenses
notebooks/153_multiconfig_optimization
notebooks/154_global_optimization_saddle_points
notebooks/155_off_axis_mirrors_nodal_aberrations
notebooks/159_phone_camera_lens_forbes_q
```

## Interferometry, holography and wavefront sensing

Measuring phase: interferometers, coherence, holography, wavefront sensors and optical testing.

- {doc}`notebooks/22_phase_unwrapping`
- {doc}`notebooks/23_interferometry`
- {doc}`notebooks/24_radius_interferometer`
- {doc}`notebooks/74_coherence_stellar_interferometry`
- {doc}`notebooks/25_shack_hartmann_vs_phase_diversity`
- {doc}`notebooks/26_holography`
- {doc}`notebooks/27_holographic_microscopy`
- {doc}`notebooks/76_schlieren_knife_edge`
- {doc}`notebooks/82_optical_coherence_tomography`

```{toctree}
:hidden:
:caption: Interferometry, holography and wavefront sensing

notebooks/22_phase_unwrapping
notebooks/23_interferometry
notebooks/24_radius_interferometer
notebooks/74_coherence_stellar_interferometry
notebooks/25_shack_hartmann_vs_phase_diversity
notebooks/26_holography
notebooks/27_holographic_microscopy
notebooks/76_schlieren_knife_edge
notebooks/82_optical_coherence_tomography
```

## Phase retrieval and beam shaping

Recovering phase from intensities, and shaping intensities with phase.

- {doc}`notebooks/28_phase_retrieval`
- {doc}`notebooks/29_TIE`
- {doc}`notebooks/30_IFTA`
- {doc}`notebooks/73_wavefront_shaping`

```{toctree}
:hidden:
:caption: Phase retrieval and beam shaping

notebooks/28_phase_retrieval
notebooks/29_TIE
notebooks/30_IFTA
notebooks/73_wavefront_shaping
```

## Microscopy

Image formation in the microscope, contrast methods, confocal and super-resolution microscopy, optical traps.

- {doc}`notebooks/52_abbe_image_formation`
- {doc}`notebooks/53_phase_contrast_dic`
- {doc}`notebooks/54_confocal_microscopy`
- {doc}`notebooks/55_fourier_ptychography`
- {doc}`notebooks/75_optical_tweezers`
- {doc}`notebooks/78_super_resolution_microscopy`
- {doc}`notebooks/27_holographic_microscopy` (in *Interferometry, holography and wavefront sensing*)

```{toctree}
:hidden:
:caption: Microscopy

notebooks/52_abbe_image_formation
notebooks/53_phase_contrast_dic
notebooks/54_confocal_microscopy
notebooks/55_fourier_ptychography
notebooks/75_optical_tweezers
notebooks/78_super_resolution_microscopy
```

## Lithography

Projection lithography from the mask to the resist: imaging models, mask models, lens and illumination design, source and mask optimization.

- {doc}`notebooks/31_vectorial_imaging_lithography`
- {doc}`notebooks/32_lithography_toolbox`
- {doc}`notebooks/33_litho_lens_aberrations`
- {doc}`notebooks/34_hopkins_socs`
- {doc}`notebooks/35_litho_resist_swing`
- {doc}`notebooks/36_litho_rigorous_mask`
- {doc}`notebooks/37_litho_mask_models`
- {doc}`notebooks/65_litho_fdtd_mask`
- {doc}`notebooks/38_litho_jones_pupil`
- {doc}`notebooks/39_litho_source_optimization`
- {doc}`notebooks/40_litho_opc`
- {doc}`notebooks/108_litho_arbitrary_layouts`
- {doc}`notebooks/41_litho_telecentricity`
- {doc}`notebooks/42_litho_illuminator`
- {doc}`notebooks/43_litho_delano_lens_design`
- {doc}`notebooks/44_seidel_caldwell_lens`
- {doc}`notebooks/50_magnification_distortion_telecentricity`
- {doc}`notebooks/107_litho_fourier_interpolation_cd`
- {doc}`150 Lens tolerances and wafer yield <notebooks/150_lens_tolerances_wafer_yield>`
- {doc}`152 Litho stochastics, flare and focus blur <notebooks/152_litho_stochastics_flare_focus_blur>`
- {doc}`156 EUV multilayers and reflective masks <notebooks/156_euv_multilayers_reflective_masks>`
- {doc}`157 EUV projector and pupil <notebooks/157_euv_projector_pupil>`
- {doc}`158 EUV stochastics and flare <notebooks/158_euv_stochastics_flare>`
- {doc}`162 Six-mirror EUV projector <notebooks/162_six_mirror_euv_projector>`

```{toctree}
:hidden:
:caption: Lithography

notebooks/31_vectorial_imaging_lithography
notebooks/32_lithography_toolbox
notebooks/33_litho_lens_aberrations
notebooks/34_hopkins_socs
notebooks/35_litho_resist_swing
notebooks/36_litho_rigorous_mask
notebooks/37_litho_mask_models
notebooks/65_litho_fdtd_mask
notebooks/38_litho_jones_pupil
notebooks/39_litho_source_optimization
notebooks/40_litho_opc
notebooks/108_litho_arbitrary_layouts
notebooks/41_litho_telecentricity
notebooks/42_litho_illuminator
notebooks/43_litho_delano_lens_design
notebooks/44_seidel_caldwell_lens
notebooks/50_magnification_distortion_telecentricity
notebooks/107_litho_fourier_interpolation_cd
notebooks/150_lens_tolerances_wafer_yield
notebooks/152_litho_stochastics_flare_focus_blur
notebooks/156_euv_multilayers_reflective_masks
notebooks/157_euv_projector_pupil
notebooks/158_euv_stochastics_flare
notebooks/162_six_mirror_euv_projector
```

## Illumination optics

Radiometry and photometry, étendue, nonimaging concentrators, microscope
illumination, homogenizers, speckle reduction, LED optics, freeform beam
shaping and the brightness of images.

- {doc}`notebooks/105_radiometry_photometry`
- {doc}`notebooks/83_etendue_brightness_theorem`
- {doc}`notebooks/84_nonimaging_concentrators`
- {doc}`notebooks/85_koehler_critical_illumination`
- {doc}`notebooks/86_homogenizers`
- {doc}`notebooks/87_speckle_reduction`
- {doc}`notebooks/88_led_optics_monte_carlo`
- {doc}`notebooks/89_freeform_beam_shaping`
- {doc}`notebooks/106_radiometry_of_images`

```{toctree}
:hidden:
:caption: Illumination optics

notebooks/105_radiometry_photometry
notebooks/83_etendue_brightness_theorem
notebooks/84_nonimaging_concentrators
notebooks/85_koehler_critical_illumination
notebooks/86_homogenizers
notebooks/87_speckle_reduction
notebooks/88_led_optics_monte_carlo
notebooks/89_freeform_beam_shaping
notebooks/106_radiometry_of_images
```

## Atmospheric optics

Rainbows, glories and the turbulent atmosphere.

- {doc}`notebooks/71_rainbows_and_glories`
- {doc}`notebooks/72_twinkling_stars_adaptive_optics`

```{toctree}
:hidden:
:caption: Atmospheric optics

notebooks/71_rainbows_and_glories
notebooks/72_twinkling_stars_adaptive_optics
```

## Photography

The optics of cameras: the out-of-focus image of a lens (bokeh), the
star its iris makes of a bright light (sunstars), the ghosts its
surfaces make of it (flare), what the sensor does to the image,
what is sharp (depth of field and tilt-shift), how a light field
camera focuses after the exposure, and how much of the blur of a phone
lens deconvolution can undo.

- {doc}`notebooks/90_bokeh`
- {doc}`notebooks/91_sunstars`
- {doc}`notebooks/92_flare_ghosts`
- {doc}`notebooks/93_sensor`
- {doc}`notebooks/94_tilt_shift_depth_of_field`
- {doc}`notebooks/95_light_field_camera`
- {doc}`160 Image restoration and PSF deconvolution <notebooks/160_image_restoration_deconvolution>`
- {doc}`161 Distortion, lateral colour and lens shading correction <notebooks/161_distortion_shading_correction>`

```{toctree}
:hidden:
:caption: Photography

notebooks/90_bokeh
notebooks/91_sunstars
notebooks/92_flare_ghosts
notebooks/93_sensor
notebooks/94_tilt_shift_depth_of_field
notebooks/95_light_field_camera
notebooks/160_image_restoration_deconvolution
notebooks/161_distortion_shading_correction
```

## Astronomy

Optics for imaging planets next to their stars: coronagraphs with
apodized pupils, occulting disks, vortex masks and Lyot stops, and a dark
hole dug with a deformable mirror; and the telescopes themselves: the
Newtonian, the classical Cassegrain, the Ritchey–Chrétien and the Schmidt
camera against the third order theory of conic mirrors, and segmented
primaries with their diffraction spikes and phasing; and gravity as a
lens: light bending traced through the index of a mass, Einstein rings,
microlensing light curves and the Einstein cross; and the echelle
spectrograph of the planet hunters, with its blaze, stacked orders and
resolving power; and the atmospheric dispersion corrector that undoes the
rainbow the air makes of every star.

- {doc}`notebooks/96_coronagraphs`
- {doc}`notebooks/97_telescope_zoo`
- {doc}`notebooks/98_segmented_mirrors`
- {doc}`notebooks/99_einstein_rings`
- {doc}`notebooks/100_echelle_spectrograph`
- {doc}`notebooks/101_atmospheric_dispersion_corrector`

```{toctree}
:hidden:
:caption: Astronomy

notebooks/96_coronagraphs
notebooks/97_telescope_zoo
notebooks/98_segmented_mirrors
notebooks/99_einstein_rings
notebooks/100_echelle_spectrograph
notebooks/101_atmospheric_dispersion_corrector
```

## Spectroscopy

Spectrometers and what they measure: prisms, gratings and the
Czerny–Turner traced with grating surfaces, grating efficiency and order
sorting; the Fourier transform spectrometer with apodization, laser
sampling, phase correction and the Jacquinot and Fellgett advantages;
Fabry–Perot etalons, interference filters and the Lyot filter; line
shapes, a model CO₂ band, the curve of growth and Beer–Lambert behind a
real instrument; and imaging spectrometers: smile and keystone of the
Offner and the Czerny–Turner, filter cameras and snapshot CASSI. The
echelle spectrograph is under Astronomy.

- {doc}`notebooks/110_grating_prism_spectrometers`
- {doc}`notebooks/111_fourier_transform_spectroscopy`
- {doc}`notebooks/113_fabry_perot_filters`
- {doc}`notebooks/114_spectral_lines_absorption`
- {doc}`notebooks/122_imaging_spectroscopy`
- {doc}`notebooks/100_echelle_spectrograph`

```{toctree}
:hidden:
:caption: Spectroscopy

notebooks/110_grating_prism_spectrometers
notebooks/111_fourier_transform_spectroscopy
notebooks/113_fabry_perot_filters
notebooks/114_spectral_lines_absorption
notebooks/122_imaging_spectroscopy
```

## AR/VR and head-up displays

The optics of augmented and virtual reality and of head-up displays:
waveguide combiners analysed in k-space, with slanted in-couplers from
RCWA, pupil replication over the eye box and the rainbow artefact; volume
holographic gratings with Kogelnik's coupled wave theory checked against
RCWA and thin films; pancake lenses that fold a VR eyepiece with
polarization, with real quartz retarders and their ghosts; and windshield
head-up displays with the double image, the wedge and a ray traced
freeform mirror.

- {doc}`123 AR waveguide combiners <notebooks/123_ar_waveguide_combiners>`
- {doc}`124 Volume holographic gratings <notebooks/124_volume_holographic_gratings>`
- {doc}`125 Pancake lenses <notebooks/125_pancake_lenses>`
- {doc}`126 Head-up displays <notebooks/126_head_up_displays>`

```{toctree}
:hidden:
:caption: AR/VR and head-up displays

notebooks/123_ar_waveguide_combiners
notebooks/124_volume_holographic_gratings
notebooks/125_pancake_lenses
notebooks/126_head_up_displays
```

## 3D sensing and LiDAR

How machines measure depth with light: the link budget of a
time-of-flight LiDAR against sunlight, detector noise and the class 1
eye safety limits that favour 1550 nm; FMCW LiDAR with its coherent
receiver, Doppler, speckle fading and the demands on the laser; the beam
scanners that sweep LiDARs over the scene; and triangulation, from laser
spot sensors and fringe projection to the dot projector of a phone's
depth camera.

- {doc}`130 Time-of-flight LiDAR <notebooks/130_time_of_flight_lidar>`
- {doc}`131 FMCW LiDAR <notebooks/131_fmcw_lidar>`
- {doc}`132 Beam scanning <notebooks/132_beam_scanning>`
- {doc}`133 Triangulation and structured light <notebooks/133_triangulation_structured_light>`

```{toctree}
:hidden:
:caption: 3D sensing and LiDAR

notebooks/130_time_of_flight_lidar
notebooks/131_fmcw_lidar
notebooks/132_beam_scanning
notebooks/133_triangulation_structured_light
```

## Surface and thin-film metrology

How fabs and optics shops measure surfaces and films: coherence scanning
(white-light) interferometry for surface heights from nanometres to
millimetres, spectroscopic ellipsometry and its model fits for film
thicknesses and optical constants, scatterometry of line gratings too
small to image, and phase measuring deflectometry for the shape and the
ripples of specular freeforms.

- {doc}`134 Coherence scanning interferometry (white light interferometry, WLI) <notebooks/134_coherence_scanning_interferometry>`
- {doc}`135 Ellipsometry <notebooks/135_ellipsometry>`
- {doc}`136 Scatterometry <notebooks/136_scatterometry>`
- {doc}`137 Deflectometry <notebooks/137_deflectometry>`

```{toctree}
:hidden:
:caption: Surface and thin-film metrology

notebooks/134_coherence_scanning_interferometry
notebooks/135_ellipsometry
notebooks/136_scatterometry
notebooks/137_deflectometry
```

## Crystal optics and light modulators

Light in anisotropic and gyrotropic media and the devices that switch,
steer and filter it: the index ellipsoid and double refraction, polarizing
prisms, optical activity and Faraday isolators, Pockels and Kerr cells and
the Mach–Zehnder modulator of every fibre link, liquid crystal cells from
the elastic theory of the director to the viewing angle of a TN display,
and acousto-optic modulators, deflectors and tunable filters. Related:
Lyot and Šolc filters with real plates in
{doc}`113 Fabry–Perot and filters <notebooks/113_fabry_perot_filters>`, a
real modulator in {doc}`120 Fibre link <notebooks/120_fibre_link>`, and an
LC tunable filter and an AOTF in
{doc}`122 Imaging spectroscopy <notebooks/122_imaging_spectroscopy>`.

- {doc}`138 Crystal optics <notebooks/138_crystal_optics>`
- {doc}`139 Electro-optic modulators <notebooks/139_electro_optic_modulators>`
- {doc}`140 Liquid crystals <notebooks/140_liquid_crystals>`
- {doc}`141 Acousto-optics <notebooks/141_acousto_optics>`

```{toctree}
:hidden:
:caption: Crystal optics and light modulators

notebooks/138_crystal_optics
notebooks/139_electro_optic_modulators
notebooks/140_liquid_crystals
notebooks/141_acousto_optics
```

## Lasers

How a laser works and what limits it: gain and saturation, the threshold
and the best output coupler, the modes of the resonator and the width of a
single one, relaxation oscillations, Q-switched and mode-locked pulses, and
the laser diode of every fibre link with its chirp and noise. Related: the
resonator in {doc}`08 ABCD matrices <notebooks/08_abcd_matrices>`, pulses
in {doc}`81 Ultrashort pulses <notebooks/81_ultrashort_pulses_lenses>`, and
the directly modulated laser of {doc}`120 Fibre link <notebooks/120_fibre_link>`.

- {doc}`142 Laser fundamentals <notebooks/142_laser_fundamentals>`
- {doc}`143 Q-switching and mode-locking <notebooks/143_q_switching_mode_locking>`
- {doc}`144 Laser diodes <notebooks/144_laser_diodes>`

```{toctree}
:hidden:
:caption: Lasers

notebooks/142_laser_fundamentals
notebooks/143_q_switching_mode_locking
notebooks/144_laser_diodes
```

## Nonlinear optics

Light that changes the medium it travels in: second harmonic generation
and phase matching in birefringent and periodically poled crystals,
parametric amplifiers and oscillators that turn one laser into a tunable
source, and the Kerr effect with self-focusing, self-phase modulation and
the Kerr lens of femtosecond lasers. Related: crystal optics in
{doc}`138 Crystal optics <notebooks/138_crystal_optics>`, mode-locked
lasers in {doc}`143 Q-switching and mode-locking <notebooks/143_q_switching_mode_locking>`,
and solitons and supercontinuum in
{doc}`121 SPM and solitons in fibres <notebooks/121_fibre_spm_solitons>`.

- {doc}`145 Second harmonic generation and phase matching <notebooks/145_second_harmonic_generation>`
- {doc}`146 Parametric amplification and OPOs <notebooks/146_parametric_amplification_opo>`
- {doc}`147 Kerr effect and self-focusing <notebooks/147_kerr_effect_self_focusing>`

```{toctree}
:hidden:
:caption: Nonlinear optics

notebooks/145_second_harmonic_generation
notebooks/146_parametric_amplification_opo
notebooks/147_kerr_effect_self_focusing
```

## Detectors and photon statistics

Where the numbers of a detector come from and what the photons
themselves do: responsivity and band gap cutoffs of Si, InGaAs and
HgCdTe, absorption depth and quantum efficiency, bandwidth against
capacitance, NEP and D* with the background limit of infrared detectors,
photon counting with SPADs and their dead time, Poisson, super- and
sub-Poissonian light, the beam splitter with photons and the
Hong–Ou–Mandel dip. Related: the receiver of a link in
{doc}`120 Fibre link <notebooks/120_fibre_link>`, SPAD histograms in
{doc}`130 Time-of-flight LiDAR <notebooks/130_time_of_flight_lidar>`,
the Fellgett advantage in
{doc}`111 Fourier transform spectroscopy <notebooks/111_fourier_transform_spectroscopy>`
and intensity interferometry in
{doc}`74 Coherence and stellar interferometry <notebooks/74_coherence_stellar_interferometry>`.

- {doc}`148 Detectors and photon statistics <notebooks/148_detectors_photon_statistics>`

```{toctree}
:hidden:
:caption: Detectors and photon statistics

notebooks/148_detectors_photon_statistics
```

## Running the notebooks

```{include} ../notebooks/README.md
:start-after: <!-- docs:running-start -->
:end-before: <!-- docs:running-end -->
```
