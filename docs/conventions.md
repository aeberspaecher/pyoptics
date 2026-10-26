# Conventions

All of pyoptics uses the time dependence exp(-iωt), which is implicit and
never written out, and positive spatial phases for propagation, i.e. fields
vary as exp(i(**k**·**r** − ωt)). In particular:

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
- Absorbing media have the complex refractive index n + iκ with κ > 0. Under
  total internal reflection, the transmitted wave decays away from the
  interface.
- Wave plates delay the slow (y) component by exp(+iδ). The Jones vector
  [1, i]/√2 rotates from x to y in time and has the Stokes parameter
  S₃ = 2 Im(E_x* E_y) = +1.
