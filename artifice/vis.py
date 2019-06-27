"""Utils for visualizing artifice output. (Mostly for testing).

TODO: make `show` functions wrappers around `plot` functions, which can be
called without clearing the matplotlib buffer.

"""

import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from artifice import img, utils

logger = logging.getLogger('artifice')


def plot_image(*images, columns=10, ticks=False, scale=20, cmap='gray'):
  cmaps = utils.listify(cmap, len(images)) 
  columns = min(columns, len(images))
  rows = max(1, len(images) // columns)
  fig, axes = plt.subplots(rows,columns, squeeze=False,
                           figsize=(scale, scale*rows/columns))
  for i, image in enumerate(images):
    ax = axes[i // columns, i % columns]
    im = ax.imshow(np.squeeze(image), cmap=cmaps[i])
  if not ticks:
    for ax in axes.ravel():
      ax.axis('off')
      ax.set_aspect('equal')
  fig.subplots_adjust(wspace=0, hspace=0)
  return fig, axes

