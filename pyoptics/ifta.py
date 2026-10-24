#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Module that implements IFTA-style algorithms.

All algorithms share common properties:

1. They need a transform from "here" to "there". "Here" will be called
object space, "there" will be called image space.
2. For each space, a method that enforces constraints needs to be given.
3. An error condition needs to be supplied.

Both the constraint enforcers and the error condition will be given these
arguments:

curr_obj, curr_img, num_iter.
"""

import numpy as np

from .fft import FT_unitary, inv_FT_unitary


img_E = lambda img: np.sum(np.abs(img)**2)
norm_to_E = lambda img, E : img*np.sqrt(E)/np.sqrt(img_E(img))


# FIXME: implement an energy rescaling step in the constrainer function? or add assert statements for the image energy in the first place?


class IFTABase(object):
    """Base class for IFTA-style algorithms.

    Derived class take everything it needs to construct a forward and a backward
    transform as the constructor's argument, build the transforms, the
    constraint enforcing routines and the error estimate and finally pass those
    routines up to the IFTA base class, which implements the iteration itself.

    This approach allows easy and quick implementation of the Gerchberg-Saxton
    algorithm as well as Fienup-style Hybrid-Input-Output algorithms.
    In principle, it should also be possible to devise classes that handle
    e.g. near-field beam shaping elements.
    """

    def __init__(self, to_image_space, to_object_space, image_space_constrainer,
                 object_space_constrainer, stopping_condition):

        self.to_image_space = to_image_space
        self.to_object_space = to_object_space
        self.image_space_constrainer = image_space_constrainer
        self.object_space_constrainer = object_space_constrainer
        self.stopping_condition = stopping_condition

    def iterate(self, start_img):
        i = 1
        curr_img = start_img.copy()
        curr_obj = self.to_object_space(np.zeros_like(start_img), start_img, np.zeros_like(start_img), curr_img, i)  # TODO: how to deal with the initial obj?
        old_obj = curr_obj.copy()
        old_img = start_img.copy()

        while(not self.stopping_condition(curr_obj, curr_img, old_obj, old_img, i)):
            old_obj = curr_obj.copy()
            old_img = curr_img.copy()

            curr_obj =  self.to_object_space(curr_obj, curr_img,  old_obj, old_img, i)
            curr_obj = self.object_space_constrainer(curr_obj, curr_img,  old_obj, old_img, i)

            curr_img = self.to_image_space(curr_obj, curr_img, old_obj, old_img, i)
            curr_img = self.image_space_constrainer(curr_obj, curr_img, old_obj, old_img, i)

            i += 1

        curr_obj =  self.to_object_space(curr_obj, curr_img,  old_obj, old_img, i)
        curr_obj = self.object_space_constrainer(curr_obj, curr_img,  old_obj, old_img, i)
        curr_img = self.to_image_space(curr_obj, curr_img, old_obj, old_img, i)

        return curr_obj, curr_img, i


class GS(IFTABase):
    """Simple Gerchberg-Saxton.
    """

    # TODO: implement smarter stopping condition!
    def __init__(self, obj_I, img_I, N_max):
        to_img = lambda curr_obj, curr_img, old_obj, old_img, i : FT_unitary(curr_obj)
        to_obj = lambda curr_obj, curr_img, old_obj, old_img, i : inv_FT_unitary(curr_img)
        stop_con = lambda curr_obj, curr_img, old_obj, old_img, i : i > N_max
        obj_enforcer = lambda curr_obj, curr_img, old_obj, old_img, i : np.sqrt(obj_I)*np.exp(1j*np.angle(curr_obj))
        img_enforcer = lambda curr_obj, curr_img, old_obj, old_img, i : np.sqrt(img_I)*np.exp(1j*np.angle(curr_img))
        super(self.__class__, self).__init__(to_img, to_obj, img_enforcer, obj_enforcer, stop_con)


#class IFTA_HIO(IFTABase):
    ## FIXME: is this all correct and still needed?

    #def _img_enforcer(self, curr_obj, curr_img, old_obj, old_img, i):
        #"""Fourier-plane to object space transform including the HIO correction
        #step.
        #"""

        ## NOTE: on entry, img is the current unconstrained object!

        #constrained_img = np.zeros_like(curr_img)

        ## generate mask of idices for which the current image is not close to the target image:
        #curr_A_img = np.abs(curr_img)
        ##mask = (curr_A_img > 1E-6) & (~self.img_support)  # current intensity non-zero outside original image's support?
        #mask = self.img_support
        ##mask = (np.abs(curr_A_img - self.img_A) > 0.1*np.max(self.img_A)) & ~self.img_support

        #delta = self.img_A*(2*np.exp(1j*np.angle(curr_img)) - np.exp(1j*np.angle(old_img))) - old_img

        ##constrained_img[~mask] = curr_img[~mask]  #self.img_A[~mask]*np.exp(1j*np.angle(curr_img[~mask]))
        ##constrained_img[mask] = curr_img[mask] + self.beta*delta[mask]
        #constrained_img = curr_img + self.beta*delta
        ##constrained_img[mask] = (old_img - self.beta*curr_img)[mask]

        ##plt.imshow(np.abs(constrained_img)**2); plt.show()
        ##plt.imshow(mask); plt.show()

        #return constrained_img


    #def __init__(self, obj_I, img_I, N_max):
        #self.beta = 0.8

        #self.obj_A = np.sqrt(obj_I)
        #self.img_A = np.sqrt(img_I)

        #self.img_support = self.img_A > 1E-12

        #to_img = lambda curr_obj, curr_img, old_obj, old_img, i : FT_unitary(curr_obj)
        #to_obj = lambda curr_obj, curr_img, old_obj, old_img, i : inv_FT_unitary(curr_img)
        #stop_con = lambda curr_obj, curr_img, old_obj, old_img, i : i > N_max
        #img_enforcer = self._img_enforcer
        #obj_enforcer = lambda curr_obj, curr_img, old_obj, old_img, i : self.obj_A*np.exp(1j*np.angle(curr_obj))
        #super(self.__class__, self).__init__(to_img, to_obj, img_enforcer, obj_enforcer, stop_con)


class IFTA_HOO(IFTABase):
    # Output-Output according to "Review of iterative Fourier-transform algorithms for beam shaping applicaitons"

    def _img_enforcer(self, curr_obj, curr_img, old_obj, old_img, i):
        """Fourier-plane to object space transform including the HIO-HOO
        correction step.
        """

        #mask = self.img_A > 0.05*np.max(self.img_A)

        delta = self.img_A*(2*np.exp(1j*np.angle(curr_img)) - np.exp(1j*np.angle(old_img))) - old_img
        #constrained_img = np.zeros_like(curr_img)
        #constrained_img[mask] = (curr_img + self.beta*delta)[mask]
        constrained_img = (curr_img + self.beta*delta)

        return constrained_img


    def __init__(self, obj_I, img_I, N_max):
        self.beta = 0.9

        self.obj_A = np.sqrt(obj_I)
        self.img_A = np.sqrt(img_I)

        to_img = lambda curr_obj, curr_img, old_obj, old_img, i : FT_unitary(curr_obj)
        to_obj = lambda curr_obj, curr_img, old_obj, old_img, i : inv_FT_unitary(curr_img)
        stop_con = lambda curr_obj, curr_img, old_obj, old_img, i : i > N_max
        img_enforcer = self._img_enforcer
        obj_enforcer = lambda curr_obj, curr_img, old_obj, old_img, i : self.obj_A*np.exp(1j*np.angle(curr_obj))
        super(self.__class__, self).__init__(to_img, to_obj, img_enforcer, obj_enforcer, stop_con)


class IFTA_OverCompensation(IFTABase):

    def _img_enforcer(self, curr_obj, curr_img, old_obj, old_img, i):
        """Fourier-plane to object space transform including the HIO correction
        step.
        """

        curr_ampl = np.abs(curr_img)
        old_ampl_sum = np.sum(curr_ampl)
        mask = (curr_ampl > 0.4*np.max(self.img_A)) &  (curr_ampl < 1.4*np.max(self.img_A))  # TODO: make adjustable
        constrained_img = np.zeros_like(curr_img)
        constrained_img[~mask] = curr_img[~mask]
        constrained_img[mask] = (np.abs(old_img)*self.img_A/curr_ampl*np.exp(1j*np.angle(curr_img)))[mask]
        constrained_img *= old_ampl_sum/np.sum(np.abs(constrained_img))

        return constrained_img


    def __init__(self, obj_I, img_I, N_max):

        self.obj_A = np.sqrt(obj_I)
        self.img_A = np.sqrt(img_I)

        self.img_support = self.img_A > 0.05*np.max(self.img_A)

        to_img = lambda curr_obj, curr_img, old_obj, old_img, i : FT_unitary(curr_obj)
        to_obj = lambda curr_obj, curr_img, old_obj, old_img, i : inv_FT_unitary(curr_img)
        stop_con = lambda curr_obj, curr_img, old_obj, old_img, i : i > N_max
        img_enforcer = self._img_enforcer
        obj_enforcer = lambda curr_obj, curr_img, old_obj, old_img, i : self.obj_A*np.exp(1j*np.angle(curr_obj))
        super(self.__class__, self).__init__(to_img, to_obj, img_enforcer, obj_enforcer, stop_con)


def digitize_phase(field, levels):
    """Rasterize phase to a given number of levels.
    Useful for e.g. multi-level beam shaping elements.
    """

    ampl = np.abs(field)
    phase = np.angle(field)

    phase_quantized = np.zeros_like(phase)
    bins = np.linspace(-np.pi, +np.pi, levels+1, endpoint=True)

    for i in range(0, levels):
        bin_left = bins[i]
        bin_right = bins[i+1]
        phase_quantized[(bin_left <= phase) & (phase < bin_right)] = (bin_left + bin_right)/2.0

    new_field = ampl*np.exp(1j*phase_quantized)

    return new_field
