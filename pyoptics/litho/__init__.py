#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Lithography imaging toolbox.

The building blocks of a lithographic imaging simulation, each one
exchangeable:

- illumination: `Illumination` sources (coherent, conventional, annular,
  dipole, quadrupole, quasar, custom maps) for any sigma and any polarization
  (x, y, TE/azimuthal, TM/radial, unpolarized, Jones or Stokes vectors,
  partially polarized via a degree of polarization),
- mask: a `MaskLayout` (geometry of one period: lines and spaces, isolated
  lines, dense and isolated contact holes, any set of rectangles) and a mask
  stack (`ThinMask`: Kirchhoff transmissions such as binary or attenuated
  PSM; `FilmMask`: transmissions of the absorber computed with
  `pyoptics.thin_films`, angle and polarization dependent), both optionally
  with a polarization dependent boundary layer at the edges (`BoundaryLayer`)
  as an approximation to thick mask effects,
- `Projector`: the projection lens (NA, wavelength, immersion, vectorial or
  scalar model, Zernike aberrations, apodization (e.g. apodization
  Zernikes), defocus),
- `WaferStack`: optional resist on a thin film stack (BARC, substrate, top
  coat), computed with `pyoptics.thin_films` for TE and TM, with standing
  waves; without it the aerial image in the immersion medium is computed,
- `LithoSimulation`: puts them together and computes images (Abbe's method,
  exact sum over diffraction orders on any rectilinear grid), focus-exposure
  matrices and process windows,
- `metrics`: contrast, ILS/NILS, threshold CDs, dose to size, Bossung
  curves, exposure latitude vs. depth of focus, MEEF,
- `lens`: aberrations of a ray traced lens (`pyoptics.raytrace`) as
  Zernikes, distortion and telecentricity per field point, CD, placement
  and best focus across a scanner slit, Zernike sensitivities,
- `hopkins`: Hopkins imaging with transmission cross coefficients (`TCC`)
  and their SOCS decomposition, exact against Abbe for thin masks and fast
  for many masks with the same optics,
- `plot`: source, mask, pupil fill, images, Bossung plots, process windows.

All lengths are image-side (wafer scale) values; mask dimensions are given
at wafer scale as well (1x), the reduction ratio of the projector is only
used to obtain the angles of incidence at the mask.

Convention: fields vary as exp(i(k.r - omega*t)).
"""

from .illumination import Illumination
from .mask import MaskLayout, ThinMask, FilmMask, BoundaryLayer
from .wafer import WaferStack
from .imaging import Projector, LithoSimulation
from . import metrics
from . import lens
from . import hopkins
