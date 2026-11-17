"""Test configuration.

Under pytest-xdist (python -m pytest -n auto test) every worker process would
otherwise start as many FFT, numba and BLAS threads as there are cores, and
the oversubscribed threads slow the run down many times over. Each worker
therefore uses a single thread; the BLAS limit has to be set before NumPy is
imported.
"""

import os

if os.environ.get("PYTEST_XDIST_WORKER"):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    import pyoptics.utils

    pyoptics.utils.set_threads(1)
