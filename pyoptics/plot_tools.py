#!/usr/bin/env python
#-*- coding:utf-8 -*-

import matplotlib.pyplot as plt


default_cmap = plt.cm.Blues


def plot_image(I, x, y, xlabel=None, ylabel=None, title=None, colorbar=True,
               **kwargs):
    """Plot an image.
    """

    if("cmap" not in kwargs):
        kwargs["cmap"] = default_cmap

    plt.imshow(I, extent=(x[0], x[-1], y[0], y[-1]), origin="lower",
               **kwargs)
    if(xlabel is not None):
        plt.xlabel(xlabel)
    if(ylabel is not None):
        plt.ylabel(ylabel)
    if(title is not None):
        plt.title(title)
    if(colorbar is not None):
        plt.colorbar()
    plt.tight_layout()
    plt.show()
