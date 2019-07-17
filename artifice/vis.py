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


def plot_image(*images, columns=10, ticks=True, scale=20, colorbar=False,
               cmap='gray', cram=False, **kwargs):
  cmaps = utils.listify(cmap, len(images)) 
  columns = min(columns, len(images))
  rows = max(1, len(images) // columns)
  fig, axes = plt.subplots(rows,columns, squeeze=False,
                           figsize=(scale, scale*rows/columns))
  for i, image in enumerate(images):
    ax = axes[i // columns, i % columns]
    if image is None:
      ax.axis('off')
      continue
    im = ax.imshow(np.squeeze(image), cmap=cmaps[i], **kwargs)
    if colorbar:
      fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
  for ax in axes.ravel():
    if not ticks:
      ax.axis('off')
    ax.set_aspect('equal')
  if cram:
    fig.subplots_adjust(wspace=0, hspace=0)
  return fig, axes

