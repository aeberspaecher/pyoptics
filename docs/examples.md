# Examples

The example notebooks illustrate the modules with examples whose physics
is easy to follow, and compare the numerics with analytic results wherever
an analytic solution exists. They are shown here with their stored outputs;
to run them yourself, see [Running the notebooks](#running-the-notebooks).

## Propagation and diffraction

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

Full-wave solutions of Maxwell's equations on the Yee grid, in the time
domain (`fdtd`) and one frequency at a time (`fdfd`). The FDTD mask
model for lithography is in {doc}`notebooks/65_litho_fdtd_mask`.

- {doc}`notebooks/57_fdtd_basics`
- {doc}`notebooks/59_fdtd_metals_plasmons`
- {doc}`notebooks/60_fdtd_photonic_circuits`
- {doc}`notebooks/62_fdtd_periodic_structures`
- {doc}`notebooks/63_fdtd_devices`
- {doc}`notebooks/64_fdtd_3d_nanojet`
- {doc}`notebooks/66_fdfd_basics`
- {doc}`notebooks/67_fdfd_limacon_microcavity`

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
```

## Beams, modes and paraxial optics

- {doc}`notebooks/06_gaussian_beams`
- {doc}`notebooks/07_beam_quality`
- {doc}`notebooks/08_abcd_matrices`
- {doc}`notebooks/09_mode_analysis`
- {doc}`notebooks/68_bessel_airy_self_healing`
- {doc}`notebooks/69_optical_vortices`

```{toctree}
:hidden:
:caption: Beams, modes and paraxial optics

notebooks/06_gaussian_beams
notebooks/07_beam_quality
notebooks/08_abcd_matrices
notebooks/09_mode_analysis
notebooks/68_bessel_airy_self_healing
notebooks/69_optical_vortices
```

## Interfaces, thin films and polarization

- {doc}`notebooks/10_interfaces_thin_films`
- {doc}`notebooks/11_polarization`
- {doc}`notebooks/12_layer_stacks`
- {doc}`notebooks/13_stack_transfer_functions`
- {doc}`notebooks/14_coated_laser_mirror`

```{toctree}
:hidden:
:caption: Interfaces, thin films and polarization

notebooks/10_interfaces_thin_films
notebooks/11_polarization
notebooks/12_layer_stacks
notebooks/13_stack_transfer_functions
notebooks/14_coated_laser_mirror
```

## Imaging, aberrations and ray tracing

- {doc}`notebooks/15_imaging_zernikes`
- {doc}`notebooks/16_bases`
- {doc}`notebooks/17_psf_mtf_strehl`
- {doc}`notebooks/19_ray_tracing`
- {doc}`notebooks/20_polarization_raytrace`
- {doc}`notebooks/21_lens_focus_waves`
- {doc}`notebooks/47_rays_and_waves`
- {doc}`notebooks/77_grin_optics`
- {doc}`notebooks/48_laser_head_doe`
- {doc}`notebooks/49_eye_aberrometer`
- {doc}`notebooks/51_fresnel_lenses`
- {doc}`notebooks/56_gaussian_beam_decomposition`
- {doc}`notebooks/58_lens_optimization`
- {doc}`notebooks/61_glass_catalogues`
- {doc}`notebooks/70_monte_carlo_tolerancing`
- {doc}`notebooks/45_zernike_aberration_gallery`
- {doc}`notebooks/79_lensless_coded_aperture_imaging`

```{toctree}
:hidden:
:caption: Imaging, aberrations and ray tracing

notebooks/15_imaging_zernikes
notebooks/16_bases
notebooks/17_psf_mtf_strehl
notebooks/19_ray_tracing
notebooks/20_polarization_raytrace
notebooks/21_lens_focus_waves
notebooks/47_rays_and_waves
notebooks/77_grin_optics
notebooks/48_laser_head_doe
notebooks/49_eye_aberrometer
notebooks/51_fresnel_lenses
notebooks/56_gaussian_beam_decomposition
notebooks/58_lens_optimization
notebooks/61_glass_catalogues
notebooks/70_monte_carlo_tolerancing
notebooks/45_zernike_aberration_gallery
notebooks/79_lensless_coded_aperture_imaging
```

## Measurement, phase retrieval and beam shaping

- {doc}`notebooks/18_image_processing`
- {doc}`notebooks/22_phase_unwrapping`
- {doc}`notebooks/23_interferometry`
- {doc}`notebooks/24_radius_interferometer`
- {doc}`notebooks/76_schlieren_knife_edge`
- {doc}`notebooks/74_coherence_stellar_interferometry`
- {doc}`notebooks/25_shack_hartmann_vs_phase_diversity`
- {doc}`notebooks/26_holography`
- {doc}`notebooks/27_holographic_microscopy`
- {doc}`notebooks/28_phase_retrieval`
- {doc}`notebooks/29_TIE`
- {doc}`notebooks/30_IFTA`
- {doc}`notebooks/73_wavefront_shaping`

```{toctree}
:hidden:
:caption: Measurement, phase retrieval and beam shaping

notebooks/18_image_processing
notebooks/22_phase_unwrapping
notebooks/23_interferometry
notebooks/24_radius_interferometer
notebooks/76_schlieren_knife_edge
notebooks/74_coherence_stellar_interferometry
notebooks/25_shack_hartmann_vs_phase_diversity
notebooks/26_holography
notebooks/27_holographic_microscopy
notebooks/28_phase_retrieval
notebooks/29_TIE
notebooks/30_IFTA
notebooks/73_wavefront_shaping
```

## Microscopy

- {doc}`notebooks/52_abbe_image_formation`
- {doc}`notebooks/53_phase_contrast_dic`
- {doc}`notebooks/54_confocal_microscopy`
- {doc}`notebooks/55_fourier_ptychography`
- {doc}`notebooks/75_optical_tweezers`
- {doc}`notebooks/78_super_resolution_microscopy`
- {doc}`notebooks/27_holographic_microscopy` (in *Measurement, phase retrieval and beam shaping*)

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
- {doc}`notebooks/41_litho_telecentricity`
- {doc}`notebooks/42_litho_illuminator`
- {doc}`notebooks/43_litho_delano_lens_design`
- {doc}`notebooks/44_seidel_caldwell_lens`
- {doc}`notebooks/50_magnification_distortion_telecentricity`

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
notebooks/41_litho_telecentricity
notebooks/42_litho_illuminator
notebooks/43_litho_delano_lens_design
notebooks/44_seidel_caldwell_lens
notebooks/50_magnification_distortion_telecentricity
```

## Atmospheric optics

- {doc}`notebooks/71_rainbows_and_glories`
- {doc}`notebooks/72_twinkling_stars_adaptive_optics`

```{toctree}
:hidden:
:caption: Atmospheric optics

notebooks/71_rainbows_and_glories
notebooks/72_twinkling_stars_adaptive_optics
```

## Running the notebooks

```{include} ../notebooks/README.md
:start-after: <!-- docs:running-start -->
:end-before: <!-- docs:running-end -->
```
