pyoptics
========

About
-----

pyoptics is a collection of tools for wave optical simulation. It contains
classes for imaging simulations and wave propagation. pyoptics has a modular
structure that easily allows reuse of individual blocks in new contexts.

Status
------

pyoptics is currently in an early pre-beta stadium.

Vague status of some of the modules:

- bases: mature
- thin_films: rather mature
- propagators: basic, missing documentation
- utils: mature
- beams: basic
- IFTA: basic
- TIE: basic
- imaging: missing completely
- processing: missing
- polarization: basic Müller calculus support, Jones calculus missing
- wpm: basic, untested

Configuration and installation
------------------------------

pyoptics is a Python package with one Cython extension (`wpm`). It is built
with setuptools via `pyproject.toml`; a C compiler is required. Cython and
NumPy are pulled in automatically at build time.

    pip install .            # regular install
    pip install -e .         # editable install for development
    pip install -e '.[plot,fftw,test]'   # with optional extras

On Linux and Windows the `wpm` extension is compiled with OpenMP. After editing
`wpm.pyx`, re-run `pip install -e .` to rebuild it.

Documentation
-------------

Documentation is currently mostly absent. Please refer to individual functions'
docstrings.

License
-------

Copyright 2014-2017 Alexander Eberspächer

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
