"""Wrapper around C functions for detecting peaks in 2D images using gradient
ascent."""

import numpy as np
cimport numpy as np

def detect_peaks():
  cdef np.ndarray[np.float32_t, ndim=2] x = np.ones((100,100), np.float32)
  return x
