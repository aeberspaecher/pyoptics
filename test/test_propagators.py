#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np

from pyoptics.propagators import fresnel_IR, fresnel_TF, suggest_N_critical, rayleigh_sommerfeld_I_TF, rayleigh_sommerfeld_I_DI, rayleigh_sommerfeld_I_IR
from pyoptics.beams import gauss_laguerre, z_r_from_divergence_angle
from pyoptics.utils import ensure_meshgrid, I


def test_beam_consistency_fresnel():
    """Test if analytical propgation of a sampled paraxial beam is close to the
    analytical solution.
    """

    wl = 0.630
    L = 3000*wl
    z_prop = 30000*wl
    N_crit = suggest_N_critical(L, z_prop, wl)
    x = np.linspace(-L/2, +L/2, N_crit)
    y = x
    w0 = 200*wl

    p, l = 3, 3
    beam_z0 = gauss_laguerre(p, l, x, y, z=0, w_0=w0,
                             z_r=z_r_from_divergence_angle(w0, np.deg2rad(0.1), 1.0, wl), wl=wl)

    beam_prop_fresnel_IR, _, _ = fresnel_IR(beam_z0, x, y, z_prop, wl)
    beam_prop_fresnel_TF, _, _ = fresnel_TF(beam_z0, x, y, z_prop, wl)
    beam_prop_analytical = gauss_laguerre(p, l, x, y, z=z_prop, w_0=w0,
                                          z_r=z_r_from_divergence_angle(w0, np.deg2rad(0.1), 1.0, wl),
                                          wl=wl)

    I_max = np.max(I(beam_prop_analytical))
    # allow largest per-pixel deviation in intensity to be 3% of maximum intensity:
    assert np.all(I(beam_prop_fresnel_IR) - I(beam_prop_analytical) < I_max/33)  # consistency of propagators
    assert np.all(I(beam_prop_fresnel_TF) - I(beam_prop_analytical) < I_max/33)  # consistency of propagators
    

def test_beam_consistency_rayleigh_sommerfeld():
    """Test if non-paraxial analytical propgation of a sampled paraxial beam is close to the
    analytical solution.
    """

    wl = 0.630
    L = 3000*wl
    z_prop = 30000*wl
    N = suggest_N_critical(L, z_prop, wl)*2
    x = np.linspace(-L/2, +L/2, N)
    y = x
    w0 = 200*wl

    p, l = 3, 3
    beam_z0 = gauss_laguerre(p, l, x, y, z=0, w_0=w0,
                             z_r=z_r_from_divergence_angle(w0, np.deg2rad(0.1), 1.0, wl), wl=wl)

    beam_prop_RS_IR, _, _ = rayleigh_sommerfeld_I_IR(beam_z0, x, y, z_prop, wl)
    beam_prop_RS_TF, _, _ = rayleigh_sommerfeld_I_TF(beam_z0, x, y, z_prop, wl)
    beam_prop_RS_DI, _, _ = rayleigh_sommerfeld_I_DI(beam_z0, x, y, z_prop, wl)
    beam_prop_analytical = gauss_laguerre(p, l, x, y, z=z_prop, w_0=w0,
                                          z_r=z_r_from_divergence_angle(w0, np.deg2rad(0.1), 1.0, wl),
                                          wl=wl)

    I_max = np.max(I(beam_prop_analytical))
    # allow largest per-pixel deviation in intensity to be 10% of maximum intensity:
    assert np.all(I(beam_prop_RS_IR) - I(beam_prop_analytical) < I_max/10)
    assert np.all(I(beam_prop_RS_TF) - I(beam_prop_analytical) < I_max/10)
    assert np.all(I(beam_prop_RS_DI) - I(beam_prop_analytical) < I_max/10)
    # TODO: 10% is arbitrary, maybe find something more justified 
 

if __name__ == '__main__':
    test_beam_consistency_fresnel()
    test_beam_consistency_rayleigh_sommerfeld()
