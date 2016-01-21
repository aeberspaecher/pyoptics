#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Routines for TIE
"""

import sympy
import numpy as np
from numpy import pi
from numpy.fft import fft, ifft, fft2, ifft2, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt



FT = lambda x: fftshift(fft2(ifftshift(x)))
inv_FT = lambda x: ifftshift(ifft2(fftshift(x)))
RMS = lambda x: 1.0/np.size(x)*np.sum(x**2)

# TODO: kick all general deriavtive stuff out of this module
# TODO: put propagation stuff in different module

def low_pass(img, x, y, sigma_cutoff, NA, lam):
    """Perform low pass filtering with frequencies that should not be
    present in the system.
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

    #plt.imshow(mask); plt.show()

    ret = np.real(inv_FT(mask*Img))

    energy_filtered = get_image_energy(ret)
    ret *= energy_orig/energy_filtered

    return ret


def low_pass_stack(stack, x, y, sigma_cutoff, NA, lam):
    ret_stack = np.zeros_like(stack)
    for i in range(np.size(stack, 2)):
        ret_stack[:,:,i] = low_pass(stack[:,:,i], x, y, sigma_cutoff, NA, lam)

    return ret_stack


def field_prop(field, x, y, delta_z, lam):
    """Propagte a complex field along z axis.
    """

    KX, KY = get_k_grid(x, y)
    k = 2*pi/lam
    kz = np.lib.scimath.sqrt(k**2 - KX**2 - KY**2)

    #if(delta_z < 0.0):
        #warning.warn("Discarding parts of transfer function!")
    H = np.exp(1j*kz*delta_z)  # transfer function

    #plt.imshow(np.abs(H), cmap=plt.cm.hot)
    #plt.colorbar()
    #plt.title("Transferfunc")
    #plt.show()

    #H = np.real(H)  # TODO: discard negative parts of transfer function

    propagated = inv_FT(H*FT(field))

    return propagated


def field_prop_paraxial(field, x, y, delta_z, lam):
    """Propagte a complex field along z axis.
    """

    KX, KY = get_k_grid(x, y)
    k = 2*pi/lam
    #kz = np.lib.scimath.sqrt(k**2 - KX**2 - KY**2)

    #if(delta_z < 0.0):
        #warning.warn("Discarding parts of transfer function!")
    H = 1 - ((KX**2 + KY**2)/k**2)*delta_z # transfer function

    #plt.imshow(np.abs(H), cmap=plt.cm.hot)
    #plt.colorbar()
    #plt.title("Transferfunc")
    #plt.show()

    #H = np.real(H)  # TODO: discard negative parts of transfer function

    propagated = inv_FT(H*FT(field))

    return propagated



def smoothed_deriv(stack, x, x0, order):
    """Compute smoothed derivatives using a Savitzky-Golay filter.
    """
    weights_sympy = sympy.calculus.finite_diff.finite_diff_weights(order, x, x0)[-1][-1]
    weights_float = [item.evalf() for item in weights_sympy]
    weights = np.asarray(weights_float, dtype=np.float64)
    #print("Weights: {}".format(weights))
    arr_stack = np.asarray(stack[:,:,:], dtype=np.float64)
    deriv = np.einsum("ijk,k->ij", arr_stack, weights)

    return deriv


def get_k_grid(x, y):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Nx, Ny = len(x), len(y)
    kx, ky = 2*pi*fftshift(fftfreq(Nx, dx)), 2*pi*fftshift(fftfreq(Ny, dy))

    return np.meshgrid(kx, ky)


def scaled_laplace_like(f, c, x, y):
    """Compute a differential operator similar to a Laplacian:
    div(c*grad(f))
    with scalar f and c.

    Use Fourier methods.
    """

    # compute x and y derivatives in k space:
    KX, KY = get_k_grid(x, y)
    F = FT(f)
    F_x, F_y = 1j*KX*F, 1j*KY*F  # TODO: sign of k values? does that match fft convention?

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
    KX, KY = get_k_grid(x, y)

    F = FT(f)
    freqs = -(KX**2 + KY**2)

    return np.real(inv_FT(freqs*F))


def inv_laplacian(f, x, y, alpha=0.0):
    """Inverse Laplacian from Fourier method using Tikhonov
    regularization.
    """

    KX, KY = get_k_grid(x, y)
    F = FT(f)
    freq_matrix = -(KX**2 + KY**2 + alpha)

    ## lift zero freqs:
    #reg_matrix = np.abs(freq_matrix) < 1E-5

    L_inv = inv_FT(F/freq_matrix)  # TODO: how to regularize the problem?

    L_inv = np.real(L_inv)  # TODO: for debugging

    return L_inv


def solve_tie(I, dI_dz, x, y, lam, alpha):
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
    """Compute dI/dz from I and a phase.
    """

    k = 2*pi/lam
    dI_dz = -1.0/k*scaled_laplace_like(phase, I, x, y)

    return dI_dz


def enhanced_tie_solve(N, I, dI_dz, x, y, lam, alpha, true_phase=None):
    # solve TIE first for the phase:
    phase0 = solve_tie(I, dI_dz, x, y, lam, alpha)

    # compute residual flux and residual phase:
    phase_improved = phase0
    dI_dz_res = dI_dz
    for i in range(N):
        print("Enanced TIE iteration {}".format(i+1))
        dI_dz_res = dI_dz - tie(phase_improved, I, x, y, lam)
        phase_res = solve_tie(I, dI_dz_res, x, y, lam, alpha)

        phase_improved = phase0 - phase_res  # TODO: sign?
        if(true_phase is not None):
            print("RMS of phase error: {}".format(RMS(true_phase-phase_improved)))

        #phase_improved = inv_laplacian(laplacian(phase_improved, x, y)
                                        #+ laplacian(phase_res, x, y),
                                       #x, y, alpha)

    # FIXME: is this routine correct?
    return phase_improved


def get_image_energy(image):
    return np.sum(image)


def energy_normalize_stack(stack, ind):
    norm_energy = get_image_energy(stack[:, :, ind])
    ret_stack = stack.copy()

    for i in range(np.shape(stack)[2]):  # iterate over all images
        curr_energy = get_image_energy(stack[:, :, i])
        ret_stack[:, :, i] *= norm_energy/curr_energy

    return ret_stack


def symmetrize_image(image):
    symmetrized = np.zeros([2*np.size(image,0), 2*np.size(image, 1)])
    Ny, Nx = np.shape(image)

    symmetrized[:Ny, :Nx] = image  # upper left part, unaltered

    symmetrized[:Ny, Nx:2*Nx] = image[:, ::-1]  # upper right
    symmetrized[Ny:2*Ny, :Nx] = image[::-1, :]  # lower left
    symmetrized[Ny:2*Ny, Nx:2*Nx] = image[::-1, ::-1]  # lower righ

    return symmetrized


def desymmetrize_image(image):
    Ny, Nx = np.shape(image)
    desymmetrized = image[:Ny//2, :Nx//2]

    return desymmetrized


def symmetrize_stack(stack, x, y):
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
    pass
#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Routines for TIE
"""

import sympy
import numpy as np
from numpy import pi
from numpy.fft import fft, ifft, fft2, ifft2, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt



FT = lambda x: fftshift(fft2(ifftshift(x)))
inv_FT = lambda x: ifftshift(ifft2(fftshift(x)))
RMS = lambda x: 1.0/np.size(x)*np.sum(x**2)

# TODO: kick all general deriavtive stuff out of this module
# TODO: put propagation stuff in different module

def low_pass(img, x, y, sigma_cutoff, NA, lam):
    """Perform low pass filtering with frequencies that should not be
    present in the system.
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

    #plt.imshow(mask); plt.show()

    ret = np.real(inv_FT(mask*Img))

    energy_filtered = get_image_energy(ret)
    ret *= energy_orig/energy_filtered

    return ret


def low_pass_stack(stack, x, y, sigma_cutoff, NA, lam):
    ret_stack = np.zeros_like(stack)
    for i in range(np.size(stack, 2)):
        ret_stack[:,:,i] = low_pass(stack[:,:,i], x, y, sigma_cutoff, NA, lam)

    return ret_stack


def field_prop(field, x, y, delta_z, lam):
    """Propagte a complex field along z axis.
    """

    KX, KY = get_k_grid(x, y)
    k = 2*pi/lam
    kz = np.lib.scimath.sqrt(k**2 - KX**2 - KY**2)

    #if(delta_z < 0.0):
        #warning.warn("Discarding parts of transfer function!")
    H = np.exp(1j*kz*delta_z)  # transfer function

    #plt.imshow(np.abs(H), cmap=plt.cm.hot)
    #plt.colorbar()
    #plt.title("Transferfunc")
    #plt.show()

    #H = np.real(H)  # TODO: discard negative parts of transfer function

    propagated = inv_FT(H*FT(field))

    return propagated


def field_prop_paraxial(field, x, y, delta_z, lam):
    """Propagte a complex field along z axis.
    """

    KX, KY = get_k_grid(x, y)
    k = 2*pi/lam
    #kz = np.lib.scimath.sqrt(k**2 - KX**2 - KY**2)

    #if(delta_z < 0.0):
        #warning.warn("Discarding parts of transfer function!")
    H = 1 - ((KX**2 + KY**2)/k**2)*delta_z # transfer function

    #plt.imshow(np.abs(H), cmap=plt.cm.hot)
    #plt.colorbar()
    #plt.title("Transferfunc")
    #plt.show()

    #H = np.real(H)  # TODO: discard negative parts of transfer function

    propagated = inv_FT(H*FT(field))

    return propagated



def smoothed_deriv(stack, x, x0, order):
    """Compute smoothed derivatives using a Savitzky-Golay filter.
    """
    weights_sympy = sympy.calculus.finite_diff.finite_diff_weights(order, x, x0)[-1][-1]
    weights_float = [item.evalf() for item in weights_sympy]
    weights = np.asarray(weights_float, dtype=np.float64)
    #print("Weights: {}".format(weights))
    arr_stack = np.asarray(stack[:,:,:], dtype=np.float64)
    deriv = np.einsum("ijk,k->ij", arr_stack, weights)

    return deriv


def get_k_grid(x, y):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Nx, Ny = len(x), len(y)
    kx, ky = 2*pi*fftshift(fftfreq(Nx, dx)), 2*pi*fftshift(fftfreq(Ny, dy))

    return np.meshgrid(kx, ky)


def scaled_laplace_like(f, c, x, y):
    """Compute a differential operator similar to a Laplacian:
    div(c*grad(f))
    with scalar f and c.

    Use Fourier methods.
    """

    # compute x and y derivatives in k space:
    KX, KY = get_k_grid(x, y)
    F = FT(f)
    F_x, F_y = 1j*KX*F, 1j*KY*F  # TODO: sign of k values? does that match fft convention?

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
    KX, KY = get_k_grid(x, y)

    F = FT(f)
    freqs = -(KX**2 + KY**2)

    return np.real(inv_FT(freqs*F))


def inv_laplacian(f, x, y, alpha=0.0):
    """Inverse Laplacian from Fourier method using Tikhonov
    regularization.
    """

    KX, KY = get_k_grid(x, y)
    F = FT(f)
    freq_matrix = -(KX**2 + KY**2 + alpha)

    ## lift zero freqs:
    #reg_matrix = np.abs(freq_matrix) < 1E-5

    L_inv = inv_FT(F/freq_matrix)  # TODO: how to regularize the problem?

    L_inv = np.real(L_inv)  # TODO: for debugging

    return L_inv


def solve_tie(I, dI_dz, x, y, lam, alpha):
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
    """Compute dI/dz from I and a phase.
    """

    k = 2*pi/lam
    dI_dz = -1.0/k*scaled_laplace_like(phase, I, x, y)

    return dI_dz


def enhanced_tie_solve(N, I, dI_dz, x, y, lam, alpha, true_phase=None):
    # solve TIE first for the phase:
    phase0 = solve_tie(I, dI_dz, x, y, lam, alpha)

    # compute residual flux and residual phase:
    phase_improved = phase0
    dI_dz_res = dI_dz
    for i in range(N):
        print("Enanced TIE iteration {}".format(i+1))
        dI_dz_res = dI_dz - tie(phase_improved, I, x, y, lam)
        phase_res = solve_tie(I, dI_dz_res, x, y, lam, alpha)

        phase_improved = phase0 - phase_res  # TODO: sign?
        if(true_phase is not None):
            print("RMS of phase error: {}".format(RMS(true_phase-phase_improved)))

        #phase_improved = inv_laplacian(laplacian(phase_improved, x, y)
                                        #+ laplacian(phase_res, x, y),
                                       #x, y, alpha)

    # FIXME: is this routine correct?
    return phase_improved


def get_image_energy(image):
    return np.sum(image)


def energy_normalize_stack(stack, ind):
    norm_energy = get_image_energy(stack[:, :, ind])
    ret_stack = stack.copy()

    for i in range(np.shape(stack)[2]):  # iterate over all images
        curr_energy = get_image_energy(stack[:, :, i])
        ret_stack[:, :, i] *= norm_energy/curr_energy

    return ret_stack


def symmetrize_image(image):
    symmetrized = np.zeros([2*np.size(image,0), 2*np.size(image, 1)])
    Ny, Nx = np.shape(image)

    symmetrized[:Ny, :Nx] = image  # upper left part, unaltered

    symmetrized[:Ny, Nx:2*Nx] = image[:, ::-1]  # upper right
    symmetrized[Ny:2*Ny, :Nx] = image[::-1, :]  # lower left
    symmetrized[Ny:2*Ny, Nx:2*Nx] = image[::-1, ::-1]  # lower righ

    return symmetrized


def desymmetrize_image(image):
    Ny, Nx = np.shape(image)
    desymmetrized = image[:Ny//2, :Nx//2]

    return desymmetrized


def symmetrize_stack(stack, x, y):
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
    pass
