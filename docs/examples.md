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
limit, fibre Bragg gratings, and a basic link with its receiver noise, eye
diagrams and budgets. Related: LP modes and fibre coupling in
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

```{toctree}
:hidden:
:caption: Fibre optics and optical communication

notebooks/115_fibre_rays_step_graded
notebooks/116_fibre_modes
notebooks/117_fibre_dispersion
notebooks/118_fibre_pulse_broadening
notebooks/119_fibre_bragg_gratings
notebooks/120_fibre_link
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

Sequential and polarization ray tracing, from rays to waves, glass catalogues, lens optimization and tolerancing, and what a lens must conserve: the Lagrange invariant, the sine and the Herschel condition, and field tracing: rays through the lenses, waves where diffraction matters.

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
what is sharp (depth of field and tilt-shift), and how a light field
camera focuses after the exposure.

- {doc}`notebooks/90_bokeh`
- {doc}`notebooks/91_sunstars`
- {doc}`notebooks/92_flare_ghosts`
- {doc}`notebooks/93_sensor`
- {doc}`notebooks/94_tilt_shift_depth_of_field`
- {doc}`notebooks/95_light_field_camera`

```{toctree}
:hidden:
:caption: Photography

notebooks/90_bokeh
notebooks/91_sunstars
notebooks/92_flare_ghosts
notebooks/93_sensor
notebooks/94_tilt_shift_depth_of_field
notebooks/95_light_field_camera
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

## Running the notebooks

```{include} ../notebooks/README.md
:start-after: <!-- docs:running-start -->
:end-before: <!-- docs:running-end -->
```
