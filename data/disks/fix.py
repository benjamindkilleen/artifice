"""Fix world-space coords from glk.
"""

from os.path import join, basename
from glob import glob
import numpy as np

shape = [100,100]
miny = 1                        # index 0
maxy = -1
minx = -1
maxx = 1

def lerp(omin, omax, imin, xx, imax):
  alpha = (xx - imin) / (imax - imin)
  return (1. - alpha)*omin + alpha*omax

def w2i(label, center="node"):
  new_label = label.copy()
  if center == "node":
    new_label[:,1] = lerp(0, shape[0] - 1, minx, label[:,0], maxx)
    new_label[:,0] = lerp(0, shape[1] - 1, miny, label[:,1], maxy)
    return new_label
  else:
    raise NotImplementedError

paths = sorted(glob("old_labels/*.txt"))
for i, path in enumerate(paths):
  if i % 500 == 0:
    print(f"{i} / {len(paths)}")
  label = np.loadtxt(path)
  label = w2i(label)
  np.savetxt(join('labels', basename(path)), label, '%.8f')
