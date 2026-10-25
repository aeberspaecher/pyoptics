API reference
=============

All modules of pyoptics, grouped by topic. Each page starts with the module
docstring, followed by summary tables and the full documentation of every
public class and function. Names in docstrings refer to the conventions in
:doc:`../conventions`.

Propagation
-----------

.. autosummary::
   :toctree: generated

   pyoptics.propagators
   pyoptics.tilted
   pyoptics.wpm
   pyoptics.fft

Beams, modes and paraxial optics
--------------------------------

.. autosummary::
   :toctree: generated

   pyoptics.beams
   pyoptics.beam_quality
   pyoptics.modes
   pyoptics.abcd

Imaging and ray tracing
-----------------------

.. autosummary::
   :toctree: generated

   pyoptics.imaging
   pyoptics.analytical
   pyoptics.bases
   pyoptics.vectorial_imaging
   pyoptics.raytrace

Lithography
-----------

.. autosummary::
   :toctree: generated

   pyoptics.litho
   pyoptics.litho.illumination
   pyoptics.litho.mask
   pyoptics.litho.wafer
   pyoptics.litho.imaging
   pyoptics.litho.metrics
   pyoptics.litho.lens
   pyoptics.litho.hopkins
   pyoptics.litho.resist
   pyoptics.litho.rcwa
   pyoptics.litho.jones_pupil
   pyoptics.litho.source_opt
   pyoptics.litho.opc
   pyoptics.litho.plot

Interfaces, thin films and polarization
---------------------------------------

.. autosummary::
   :toctree: generated

   pyoptics.interfaces
   pyoptics.thin_films
   pyoptics.coatings
   pyoptics.materials
   pyoptics.polarization
   pyoptics.polarization.jones
   pyoptics.polarization.mueller

Measurement and phase retrieval
-------------------------------

.. autosummary::
   :toctree: generated

   pyoptics.interferometry
   pyoptics.holography
   pyoptics.unwrap
   pyoptics.phase_retrieval
   pyoptics.TIE
   pyoptics.ifta
   pyoptics.processing

Utilities
---------

.. autosummary::
   :toctree: generated

   pyoptics.utils
   pyoptics.geometry
   pyoptics.masks
   pyoptics.interpolator
   pyoptics.plot_tools
