#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Routines for TIE

Convention: fields vary as exp(i(kz - omega*t)) and are written as
sqrt(I)*exp(i*phase). The transport of intensity equation then reads ::

    k dI/dz = -div(I grad(phase)),

so a converging wave (phase decreasing away from the axis, e.g. a thin lens
exp(-i k r**2/(2f))) has dI/dz > 0 on axis.

The module contains no propagator of its own: to compute defocused images, use
pyoptics.propagators, e.g. rayleigh_sommerfeld_I_TF(), which follows the same
convention.
"""

import numpy as np
from numpy import pi
from numpy.fft import fftfreq, fftshift

from .fft import FT, inv_FT


RMS = lambda x: np.sqrt(1.0/np.size(x)*np.sum(np.abs(x)**2))

# TODO: kick all general deriavtive stuff out of this module

def low_pass(img, x, y, sigma_cutoff, NA, lam):
    """Remove spatial frequencies that the imaging system cannot transmit.

    All frequencies beyond sigma_cutoff times the cut-off k*NA are set to zero
    (hard filter); the result is rescaled to the energy (sum) of the input.

    Parameters
    ----------
    img : 2d array
        Intensity image.
    x, y : 1d arrays
        Coordinates of the image (equidistant).
    sigma_cutoff : number
        Cut-off in units of the coherent cut-off frequency k*NA.
    NA : number
        Numerical aperture.
    lam : number
        Wavelength.

    Returns
    -------
    filtered : 2d array
        Filtered (real) image with the energy of img.
    """

    energy_orig = get_image_energy(img)

    Img = FT(img)
    k = 2*pi/lam

    KX, KY = get_k_grid(x, y)
    k_mesh = np.sqrt(KX**2 + KY**2)
    sigma = k_mesh/(k*NA)

    # implement hard low-pass filter: cutoff everything beyond a given sigma:
    mask = np.ones(np.shape(KX))
    mask[sigma > sigma_cutoff] = 0.0

    ret = np.real(inv_FT(mask*Img))

    energy_filtered = get_image_energy(ret)
    ret *= energy_orig/energy_filtered

    return ret


def low_pass_stack(stack, x, y, sigma_cutoff, NA, lam):
    """Apply `low_pass()` to each image of a stack.

    Parameters
    ----------
    stack : 3d array
        Image stack of shape (Ny, Nx, N_images).
    x, y, sigma_cutoff, NA, lam
        See `low_pass()`.

    Returns
    -------
    filtered : 3d array
        Filtered stack, same shape as stack.
    """

    ret_stack = np.zeros_like(stack)
    for i in range(np.size(stack, 2)):
        ret_stack[:,:,i] = low_pass(stack[:,:,i], x, y, sigma_cutoff, NA, lam)

    return ret_stack


def smoothed_deriv(stack, x, x0, order):
    """Derivative along the stack axis from finite difference weights.

    The weights for the derivative of the given order at x0 from the samples
    at x are computed with sympy (optional dependency) and applied to all
    pixels.

    Parameters
    ----------
    stack : 3d array
        Image stack of shape (Ny, Nx, len(x)).
    x : sequence of numbers
        Positions (e.g. defocus values z) of the images in the stack.
    x0 : number
        Position at which to evaluate the derivative.
    order : int
        Order of the derivative.

    Returns
    -------
    deriv : 2d array
        Derivative at x0, shape (Ny, Nx).
    """
    import sympy  # optional dependency, only needed here

    weights_sympy = sympy.calculus.finite_diff.finite_diff_weights(order, x, x0)[-1][-1]
    weights_float = [item.evalf() for item in weights_sympy]
    weights = np.asarray(weights_float, dtype=np.float64)
    arr_stack = np.asarray(stack[:,:,:], dtype=np.float64)
    deriv = np.einsum("ijk,k->ij", arr_stack, weights)

    return deriv


def get_k_grid(x, y):
    """Angular frequency grid for centered FFTs.

    Parameters
    ----------
    x, y : 1d arrays
        Equidistant coordinates.

    Returns
    -------
    KX, KY : 2d arrays
        k_x and k_y (2*pi times the spatial frequencies) on a meshgrid, centered
        as returned by fftshift().
    """

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Nx, Ny = len(x), len(y)
    kx, ky = 2*pi*fftshift(fftfreq(Nx, dx)), 2*pi*fftshift(fftfreq(Ny, dy))

    return np.meshgrid(kx, ky)


def scaled_laplace_like(f, c, x, y):
    """Compute div(c*grad(f)) with Fourier derivatives.

    For constant c this is c times the Laplacian of f.

    Parameters
    ----------
    f : 2d array
        Scalar function.
    c : 2d array or number
        Scalar weight applied to the gradient.
    x, y : 1d arrays
        Equidistant coordinates.

    Returns
    -------
    U : 2d array
        div(c*grad(f)) (real part).
    """

    # compute x and y derivatives in k space:
    KX, KY = get_k_grid(x, y)
    F = FT(f)
    F_x, F_y = 1j*KX*F, 1j*KY*F  # FT() has the kernel exp(-i k x), so d/dx -> i*k_x

    # go back to position space to scale the gradient with c:
    g_x, g_y = inv_FT(F_x), inv_FT(F_y)

    g_x, g_y = np.real(g_x), np.real(g_y)  # TODO: debugging

    h_x, h_y = c*g_x, c*g_y

    # compute divergence of h field in k space:
    H_x, H_y = FT(h_x), FT(h_y)
    U = inv_FT(1j*KX*H_x + 1j*KY*H_y)

    U = np.real(U)  # TODO: debugging

    return U


def laplacian(f, x, y):
    """Laplacian of f with Fourier derivatives.

    Parameters
    ----------
    f : 2d array
        Scalar function.
    x, y : 1d arrays
        Equidistant coordinates.

    Returns
    -------
    L : 2d array
        Laplacian of f (real part).
    """

    KX, KY = get_k_grid(x, y)

    F = FT(f)
    freqs = -(KX**2 + KY**2)

    return np.real(inv_FT(freqs*F))


def inv_laplacian(f, x, y, alpha=0.0):
    """Inverse Laplacian with Fourier methods and Tikhonov regularization.

    Divides the spectrum by -(k_x**2 + k_y**2 + alpha). Without
    regularization, the zero frequency is undefined and set to zero, i.e.
    the mean free solution is returned.

    Parameters
    ----------
    f : 2d array
        Scalar function.
    x, y : 1d arrays
        Equidistant coordinates.
    alpha : number, optional
        Regularization parameter (in units of k**2).

    Returns
    -------
    L_inv : 2d array
        Inverse Laplacian of f (real part).
    """

    KX, KY = get_k_grid(x, y)
    F = FT(f)
    freq_matrix = -(KX**2 + KY**2 + alpha)

    # without regularization, the zero frequency component is undefined; set
    # it to zero, i.e. return the mean free solution:
    zero_freq = (freq_matrix == 0.0)
    freq_matrix[zero_freq] = 1.0
    F_inv = F/freq_matrix
    F_inv[zero_freq] = 0.0

    L_inv = inv_FT(F_inv)  # TODO: how to regularize the problem?

    L_inv = np.real(L_inv)  # TODO: for debugging

    return L_inv


def solve_tie(I, dI_dz, x, y, lam, alpha):
    """Solve the transport of intensity equation for the phase.

    Inverts k dI/dz = -div(I grad(phase)) with two inverse Laplacians (see
    the module docstring for the sign convention).

    Parameters
    ----------
    I : 2d array
        Intensity in focus.
    dI_dz : 2d array
        Axial derivative of the intensity.
    x, y : 1d arrays
        Equidistant coordinates.
    lam : number
        Wavelength.
    alpha : number
        Regularization parameter, see `inv_laplacian()`.

    Returns
    -------
    phase : 2d array
        Recovered phase in radians (mean free).
    """

    k = 2*pi/lam

    iL_flow = inv_laplacian(dI_dz, x, y, alpha)

    # non_constant I:
    tmp = scaled_laplace_like(iL_flow, 1.0/I, x, y)
    ret = -k*inv_laplacian(tmp, x, y, alpha)

    ## FIXME: testcase
    ## constant I:
    #ret = -k*iL_flow/np.mean(I)  # mean: catch almost constant energy...

    # as long as we use FFT instead of RFFT, remove imaginary parts:
    ret = np.real(ret)

    return ret


def tie(phase, I, x, y, lam):
    """Forward transport of intensity equation.

    Compute dI/dz = -div(I grad(phase))/k from I and a phase (see module
    docstring for the sign convention).

    Parameters
    ----------
    phase : 2d array
        Phase in radians.
    I : 2d array
        Intensity.
    x, y : 1d arrays
        Equidistant coordinates.
    lam : number
        Wavelength.

    Returns
    -------
    dI_dz : 2d array
        Axial derivative of the intensity.
    """

    k = 2*pi/lam
    dI_dz = -1.0/k*scaled_laplace_like(phase, I, x, y)

    return dI_dz


def enhanced_tie_solve(N, I, dI_dz, x, y, lam, alpha, true_phase=None):
    # solve TIE first for the phase:
    """Solve the TIE iteratively, correcting the residual flux.

    Starts with `solve_tie()`, then repeats N times: compute the flux of the
    current phase with `tie()`, solve for the phase of the residual flux and
    add it to the estimate.

    Parameters
    ----------
    N : int
        Number of refinement iterations.
    I, dI_dz, x, y, lam, alpha
        See `solve_tie()`.
    true_phase : 2d array, optional
        Known phase; if given, the RMS error is printed after each iteration.

    Returns
    -------
    phase : 2d array
        Refined phase in radians.
    """

    phase0 = solve_tie(I, dI_dz, x, y, lam, alpha)

    # compute residual flux and residual phase:
    phase_improved = phase0
    dI_dz_res = dI_dz
    for i in range(N):
        dI_dz_res = dI_dz - tie(phase_improved, I, x, y, lam)
        phase_res = solve_tie(I, dI_dz_res, x, y, lam, alpha)

        # the residual phase solves the TIE for the residual flux, so it adds
        # to the current estimate:
        phase_improved = phase_improved + phase_res
        if true_phase is not None:
            print("RMS of phase error: {}".format(RMS(true_phase-phase_improved)))

    # FIXME: is this routine correct?
    return phase_improved


def get_image_energy(image):
    """Energy of an intensity image, i.e. the sum of its pixels.

    Parameters
    ----------
    image : array
        Intensity image.

    Returns
    -------
    energy : number
    """

    return np.sum(image)


def energy_normalize_stack(stack, ind):
    """Scale all images of a stack to the energy of one of them.

    Parameters
    ----------
    stack : 3d array
        Image stack of shape (Ny, Nx, N_images).
    ind : int
        Index of the reference image.

    Returns
    -------
    normalized : 3d array
        Copy of the stack with equal energies.
    """

    norm_energy = get_image_energy(stack[:, :, ind])
    ret_stack = stack.copy()

    for i in range(np.shape(stack)[2]):  # iterate over all images
        curr_energy = get_image_energy(stack[:, :, i])
        ret_stack[:, :, i] *= norm_energy/curr_energy

    return ret_stack


def symmetrize_image(image):
    """Antisymmetric extension of an image to twice its size.

    The image is mirrored with a sign change across its right and lower edges
    (upper left: image, upper right and lower left: -mirror, lower right:
    double mirror), which removes the jumps at the borders for FFT based
    solvers.

    Parameters
    ----------
    image : 2d array
        Image of shape (Ny, Nx).

    Returns
    -------
    symmetrized : 2d array
        Image of shape (2*Ny, 2*Nx).
    """

    symmetrized = np.zeros([2*np.size(image,0), 2*np.size(image, 1)])
    Ny, Nx = np.shape(image)

    symmetrized[:Ny, :Nx] = image  # upper left part, unaltered

    symmetrized[:Ny, Nx:2*Nx] = -image[:, ::-1]  # upper right
    symmetrized[Ny:2*Ny, :Nx] = -image[::-1, :]  # lower left
    symmetrized[Ny:2*Ny, Nx:2*Nx] = image[::-1, ::-1]  # lower right

    return symmetrized


def desymmetrize_image(image):
    """Upper left quarter of an image, the inverse of `symmetrize_image()`.

    Parameters
    ----------
    image : 2d array
        Symmetrized image of shape (2*Ny, 2*Nx).

    Returns
    -------
    image : 2d array
        Image of shape (Ny, Nx).
    """

    Ny, Nx = np.shape(image)
    desymmetrized = image[:Ny//2, :Nx//2]

    return desymmetrized


def symmetrize_stack(stack, x, y):
    """Apply `symmetrize_image()` to each image of a stack.

    Parameters
    ----------
    stack : 3d array
        Image stack of shape (Ny, Nx, N_images).
    x, y : 1d arrays
        Equidistant coordinates of the images.

    Returns
    -------
    symmetrized : 3d array
        Stack of shape (2*Ny, 2*Nx, N_images).
    x_new, y_new : 1d arrays
        Coordinates of the symmetrized images, starting at x[0] and y[0].
    """

    symmetrized = np.zeros([2*np.size(stack,0), 2*np.size(stack, 1), np.size(stack,2)])
    for i in range(np.size(stack, 2)):
        symmetrized[:,:,i] = symmetrize_image(stack[:,:,i])

    dx, dy = x[1] - x[0], y[1] - y[0]
    Nx = len(x)
    Ny = len(y)
    x_new = x[0] + np.linspace(0, 2*(Nx)*dx, 2*Nx, endpoint=False)
    y_new = y[0] + np.linspace(0, 2*(Ny)*dy, 2*Ny, endpoint=False)

    return symmetrized, x_new, y_new


def desymmetrize_stack(stack, x, y):
    """Inverse of `symmetrize_stack()` (not implemented yet, returns None).

    Parameters
    ----------
    stack : 3d array
        Symmetrized stack.
    x, y : 1d arrays
        Coordinates of the symmetrized stack.
    """

    pass
