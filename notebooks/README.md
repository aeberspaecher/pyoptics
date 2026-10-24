# Example notebooks

These Jupyter notebooks illustrate the individual parts of pyoptics with
examples whose physics is easy to follow, and visualize them extensively.
Wherever an analytic solution exists, the numerics are compared with it. The
notebooks are stored with their outputs, so the figures can be seen without
running anything (e.g. directly on GitHub).

| Notebook | Modules | Contents |
|---|---|---|
| [01_propagation](01_propagation.ipynb) | `propagators`, `analytical`, `masks` | Fresnel number and diffraction at a square (with Fresnel integrals), Talbot carpet, limits of the Fresnel approximation, evanescent waves, wrap-around and padding, transfer function vs. impulse response, band-limited propagation, Airy disk, on-axis intensity (Mahajan), Poisson spot |
| [02_gaussian_beams](02_gaussian_beams.ipynb) | `beams`, `propagators`, `masks.thin_lens` | Beam radius, curvature and Gouy phase, analytic vs. numerical, Gouy phase of higher modes, HG and LG mode gallery, rotating mode superposition, focus of a lens |
| [03_interfaces_thin_films](03_interfaces_thin_films.ipynb) | `interfaces`, `thin_films` | Fresnel coefficients, Brewster angle, total internal reflection with the evanescent wave, metal reflection, soap film colours, anti-reflection coating, Bragg mirror, coherent vs. incoherent thick plates, thin metal films |
| [04_polarization](04_polarization.ipynb) | `polarization.jones`, `polarization.mueller` | Polarization ellipses and Stokes parameters, Malus' law and polarizer chains, wave plates on the Poincaré sphere, depolarization, polarization by reflection, Fresnel rhomb, metal reflection |
| [05_imaging_zernikes](05_imaging_zernikes.ipynb) | `imaging`, `bases`, `propagators` | Fringe Zernike gallery, PSF with aberrations, Strehl ratio and Maréchal approximation, Zernike fit, coherent vs. incoherent imaging (Siemens star, CTF/OTF), Rayleigh criterion, 3D focus |
| [06_TIE](06_TIE.ipynb) | `TIE`, `propagators.rayleigh_sommerfeld_I_TF`, `utils.get_z_derivative` | Making phase objects visible by defocus, TIE forwards and backwards, noise vs. defocus distance, regularization, Savitzky-Golay derivative, iterative refinement |
| [07_IFTA](07_IFTA.ipynb) | `ifta` | Beam shaping with Gerchberg-Saxton, convergence, hybrid output-output, influence of the start phase, phase quantization and the twin image |
| [08_WPM](08_WPM.ipynb) | `wpm` | Wave propagation method: test in a homogeneous medium, GRIN medium, step-index waveguide, thick cylindrical lens with spherical aberration |
| [09_bases](09_bases.ipynb) | `bases`, `masks` | Zernike pyramid and radial polynomials, orthogonality on the grid (Gram matrices), least squares vs. overlap integrals, Legendre and Chebyshev bases on rectangles, coefficient drift on non-native apertures, numerical orthogonalization on annulus and hexagon, rescaling, fits to scattered data, presampling |
| [10_layer_stacks](10_layer_stacks.ipynb) | `thin_films`, `materials` | Interactive layer stack designer (ipywidgets): R, T, A vs. wavelength and angle for s, p and unpolarized light; AR coatings, Bragg mirror, Fabry-Perot filter, metal film; Abbe dispersion model (n_d, V_d) compared with Sellmeier, dispersive Bragg mirror and AR coating |
| [11_image_processing](11_image_processing.ipynb) | `processing`, `utils.center_of_mass`, `utils.add_camera_noise` | Scaling by field strength or intensity (peak, mean, power; per image or per stack), Poisson vs. Gaussian noise, photon transfer curve and SNR (EMVA 1288 camera model), fixed pattern noise and frame averaging, bit depth and saturation, centering jittered stacks, background and threshold, centroid precision vs. photon number, complete measurement pipeline |
| [12_psf_mtf_strehl](12_psf_mtf_strehl.ipynb) | `imaging`, `analytical` | Airy pattern and encircled energy, MTF of clear and obscured pupils, defocus and contrast reversal (chirped bar target), Strehl ratio vs. Maréchal approximations for several aberrations, apodization |
| [13_beam_quality](13_beam_quality.ipynb) | `beam_quality`, `propagators`, `processing` | ISO 11146 widths and principal axes of elliptical and Hermite-Gaussian beams, M² of a focused beam with spherical aberration (two-step caustic measurement, focal shift), M² from a single plane vs. ISO window, noise and background: why ISO iterates a 3× window |
| [14_stack_transfer_functions](14_stack_transfer_functions.ipynb) | `thin_films.stack_transfer_functions` | Phase of r at total internal reflection, Goos-Hänchen shift vs. Artmann's formula, frustrated total internal reflection (tunnelling), focused beam on a Fabry-Perot etalon: Haidinger rings and reduced peak transmission |
| [15_vectorial_diffraction](15_vectorial_diffraction.ipynb) | `propagators` (vectorial) | Longitudinal field of tilted plane waves and tight Gaussian waists, radially and azimuthally polarized beams, vectorized Fresnel and Fraunhofer propagation, padding vector fields, high NA focusing with Mansuripur's method vs. Richards-Wolf, focal spots for linear, circular, radial and azimuthal polarization, through focus, energy conservation |
| [16_vectorial_imaging_lithography](16_vectorial_imaging_lithography.ipynb) | `vectorial_imaging`, `polarization` | Two-beam interference of TE and TM waves, L/S masks and illumination sources (dipole, annular, quasar) with polarization in the pupil, aerial images of 40 nm L/S at NA 1.35 for TE, TM and unpolarized light vs. the scalar model, contrast and NILS through pitch, dry vs. immersion, defocus, partially polarized illumination (IPS), attenuated PSM, contact holes with azimuthal and radial polarization |

All notebooks use the pyoptics convention $\exp(i(\mathbf{k}\cdot\mathbf{r} - \omega t))$
(see the main README).

## Running

You need an installation of pyoptics as described in the main README, plus
matplotlib, Jupyter and (for notebook 10) ipywidgets, all contained in the
`notebooks` extra:

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/
```

The widgets of notebook 10 only work with a running kernel; static viewers
such as GitHub show the static figures stored below each interactive cell.

Run times on an ordinary laptop range from a few seconds to about a minute per
notebook (longest: 01 and 08).

To re-execute all notebooks and store their outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
