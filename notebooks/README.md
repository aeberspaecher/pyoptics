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

All notebooks use the pyoptics convention $\exp(i(\mathbf{k}\cdot\mathbf{r} - \omega t))$
(see the main README).

## Running

You need an installation of pyoptics as described in the main README, plus
matplotlib and Jupyter:

```bash
pip install matplotlib jupyterlab
jupyter lab notebooks/
```

Run times on an ordinary laptop range from a few seconds to about a minute per
notebook (longest: 01 and 08).

To re-execute all notebooks and store their outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
