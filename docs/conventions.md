# Conventions

All of pyoptics uses the time dependence exp(-iωt), which is implicit and
never written out, and positive spatial phases for propagation, i.e. fields
vary as exp(i(**k**·**r** − ωt)). Notebook
{doc}`notebooks/00_foundations_conventions` derives these conventions from
Maxwell's equations and checks each of them numerically. In particular:

- A plane wave travelling in +z is exp(+ikz). Propagating a field by a
  distance z > 0 multiplies each of its plane wave components by
  exp(+ik_z z); evanescent components decay as exp(−|k_z| z).
- `fft.FT()` has the kernel exp(−ik_x x) and returns the amplitudes of the
  plane waves exp(+ik_x x); `fft.inv_FT()` sums them up. Both work on
  centered arrays (x = 0 and k = 0 at index N//2, as `utils.freq_grid()`
  returns them), see the docstring of `pyoptics.fft`.
- A diverging spherical wave is exp(+ikr)/r. Beams (`beams.gauss_laguerre()`,
  `beams.gauss_hermite()`) carry the phase k z + k r²/(2R(z)) − ψ(z) with the
  Gouy phase ψ(z) = (N + 1) arctan(z/z_R); Laguerre-Gaussian modes have the
  helical phase exp(+ilφ).
- A converging thin lens of focal length f > 0 has the transmission
  exp(−ik r²/(2f)) (`masks.thin_lens()`).
- ABCD matrices (`abcd`) act on rays (y, nθ) with the reduced angle nθ, so
  det M = 1. Radii are positive if the centre of curvature lies in +z,
  concave mirrors have R > 0. The complex beam parameter is the textbook
  q = z + i z_R with 1/q = 1/R − iλ₀/(π n w²); the field is
  ∝ exp(+ik r²/(2R)) exp(−r²/w²) and the Gouy phase enters as exp(−iψ).
- Ray tracing (`raytrace`) uses the same radius convention; a concave mirror
  facing the incoming light has R < 0 there (real geometry, not unfolded),
  and thicknesses behind mirrors are negative. Tilts are rotations
  R_x(a_x) R_y(a_y) R_z(a_z), counterclockwise, after the decentre. The OPD is
  OPL_ray − OPL_chief on the reference sphere, so the pupil phase is
  +2π·OPD/λ: an image plane behind the focus gives OPD < 0 at the rim, as
  the propagators' exp(+ik_z z).
- FDTD (`fdtd`) works in normalized units (c = ε₀ = μ₀ = 1, lengths and
  wavelengths in the same unit). Its DFT monitors return amplitudes in the
  same convention, F = Σ f(t) exp(+iωt) Δt, so a wave towards +x has the
  phase exp(+ikx). In 2D, `mode="Ez"` has E along z (out of the plane)
  and `mode="Hz"` has H along z. The FDTD literature calls Ez "TM";
  waveguide, grating and scattering texts, `litho.rcwa` and
  `analytical.cylinder_field` call the same case "TE" (E perpendicular to
  the plane of incidence).
- FDFD (`fdfd`) shares the units, grid, shapes and media of the FDTD and
  uses exp(−iωt): outgoing waves go as exp(+ikr) and resonances have
  Im ω < 0 with Q = Re ω/(−2 Im ω), as `analytical.disk_resonance`.
- Absorbing media have the complex refractive index n + iκ with κ > 0. Under
  total internal reflection, the transmitted wave decays away from the
  interface.
- Wave plates delay the slow (y) component by exp(+iδ). The Jones vector
  [1, i]/√2 rotates from x to y in time and has the Stokes parameter
  S₃ = 2 Im(E_x* E_y) = +1.
- A positive phase is a delay: the phase is +k₀ times the optical path. A
  wavefront that lags behind a reference (a longer optical path, OPD > 0)
  has a positive phase.
- Pupils: `imaging`, `litho` and `raytrace.System.pupil()` use the
  directions of the image space plane waves, (P_x, P_y) = **k**⊥/(k NA);
  `raytrace.System.opd()` and `System.zernike()` use positions in the
  entrance pupil. For a lens that forms a real image the two are
  point-mirrored, so Zernikes odd in the azimuth (Z2, Z3, Z7, Z8, ...)
  change sign between them. Zernikes are Fringe Zernikes (University of
  Arizona order, 1 at the rim), the azimuth runs from +x towards +y, and
  a positive Z4 puts the best focus behind the plane of evaluation.
- Fresnel coefficients: `interfaces` gives ratios of the full fields with
  Hecht's reference directions (r_p = −r_s at normal incidence);
  `thin_films` and `coatings` give ratios of the tangential fields (r_p =
  r_s at normal incidence, t_p larger by cos θ₂/cos θ₁). R and T agree.
- Ellipsometry (`ellipsometry`, `scatterometry`) reports Ψ and Δ in the
  Nebraska conventions of the ellipsometry literature (Born & Wolf's r_p
  and exp(+iωt)): ρ = tan Ψ exp(iΔ) is the complex conjugate of
  −r_p/r_s with the tangential r_p of `thin_films`, so bare silicon has Δ
  just below 180° and a native oxide lowers it. Optical constants stay
  n + iκ, and `ellipsometry.pseudo_dielectric()` returns ε₁ + iε₂.
- Coherence scanning interferometry (`csi`) follows the pyoptics sign: a
  lower surface has the longer path and the larger phase, the fringe term
  is Re(r_S exp(2ik cos θ (z − h))) for the scan position z, and frequency
  domain analysis returns the phase K h − arg r_S.
- Crystal optics (`crystals`): the 4×4 method uses ψ = (Ex, Hy, Ey, Hx)
  with dψ/dz = i k₀ Δ ψ; Jones matrices are in the (p, s) basis, or in
  the lab x-y basis when an azimuth is given, with amplitudes as lengths
  of E (powers need `power_factor()`). A rotation angle is counted
  counterclockwise from x towards y looking towards the source;
  right-handed quartz rotates by −ρd. Faraday rotation keeps its sense
  in the lab for both directions of travel, natural activity reverses.
- Modulators (`electrooptics`) follow `fibrecomm`: a chirp parameter
  α > 0 means blue-shifted rising edges, like a directly modulated laser,
  and the instantaneous frequency shift is −dφ/dt for exp(−iωt).
- Lasers (`lasers`) use SI units, except `DiodeLaser.nrz()`, which returns
  its field in the units of `fibrecomm` (ps, √mW). Its chirp has the same
  sign as above: a positive linewidth enhancement factor α_H gives
  blue-shifted rising edges. The rate equations are normalized to the
  threshold inversion, and the pump r to its threshold value.
- Nonlinear optics (`nonlinear`) uses SI units with Boyd's field
  convention E = A exp(i(kz − ωt)) + c.c. (the pyoptics sign), amplitudes
  normalized to the intensity (|a|² = I), the phase mismatch
  Δk = k₁ + k₂ − k₃ (2k₁ − k₂ for SHG, + 2πm/Λ for quasi-phase matching)
  and wavelengths in nm where a name says `wavelength_nm`; d coefficients
  in `CRYSTALS` are in pm/V. Pulses (`splitstep`) follow `fibrecomm`:
  spectra A~(Ω) = ∫A exp(iΩt) dt, positive Ω is a higher frequency, the
  dispersion operator is i Σ β_k Ω^k/k! − α/2 and the Kerr phase is
  +γ|A|²z, so that SPM shifts the rising edge to the red. Units are free
  but consistent (ps, km, W in the fibre notebooks).
- Detectors (`detectors`) use SI units (W, A, Hz, m, s), wavelengths in
  nm where a name says `wavelength_nm`, band gaps in eV and D* in Jones
  (cm Hz^½/W). `fibrecomm` wraps its receiver functions in mW, mA and
  GHz. The beam splitter is symmetric, t = √T and r = i√R.
- Circular polarization is named about the direction of propagation (IEEE):
  `jones.J_RHCP` = [1, i]/√2 has positive helicity. Born & Wolf and Hecht
  call the same field left-circular.
- "Intensity": the wave optics modules and notebooks follow the wave
  optics habit and call |E|² the intensity, a power per area (an
  irradiance in radiometric terms, up to a constant). In `radiometry`,
  `illumination` and notebooks 88 (LED optics), 105 (radiometry and photometry) and 106 (radiometry of images), intensity has its
  radiometric meaning, power per solid angle (W/sr, or cd); notebook 105 (radiometry and photometry)
  defines flux, intensity, irradiance and radiance.
- Arrays of the wave optics modules are indexed [y, x] (from
  `utils.grid2d()`, plotted with `origin="lower"`); the FDTD and FDFD index
  their fields [x, y] (or [x, y, z]).
