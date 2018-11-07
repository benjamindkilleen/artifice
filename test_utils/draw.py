"""Drawing functions for use in experiments.py. These functions are very
similar, in terms of arguments and return values, as skimage.draw.
"""

import numpy as np

def satisfying(func, shape):
  """Return the (i,j) indices of image-space that satisfy boolean func(i,j),
  within an image with `shape`.

  If either element of shape is a len 2 array-like, then treat it as [lo, hi)
  TODO: this.

  Indices returned in two numpy arrays, rr, cc, which can be used to index the
  image as img[rr,cc] = val.
  """

  assert(len(shape) == 2)
  real_shape = np.zeros((2,2), dtype=int)
  for i in range(len(shape)):
    dim = shape[i]
    if hasattr(dim, '__len__'):
      assert(len(dim == 1))
      real_shape[i,0] = int(dim[0])
      real_shape[i,1] = int(dim[1])
    else:
      real_shape[i,1] = int(dim)

  xs = []
  ys = []
  
  for x in range(real_shape[0,0], real_shape[0,1]):
    for y in range(real_shape[1,0], real_shape[1,1]):
      if func(x,y):
        xs.append(x)
        ys.append(y)

  return np.array(xs), np.array(ys)

def circle(center, radius, shape=None):
  """Return the skimage.draw-like indices of the circle. If shape != None, then
  limit arrays to those inside shape.
  """
  center = np.array(center)
  assert(center.shape == (2,))
  if shape == None:
    shape = np.array(
      [[int(center[0] - radius - 1), int(center[0] + radius + 1)],
       [int(center[1] - radius - 1), int(center[1] + radius + 1)]])

  func = lambda x,y : np.sqrt((x-center[0])**2 + (y-center[1])**2) < radius
  return satisfying(func, shape)

