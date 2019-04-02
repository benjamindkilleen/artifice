"""Utils for visualizing artifice output. (Mostly for testing).

TODO: make `show` functions wrappers around `plot` functions, which can be
called without clearing the matplotlib buffer.

"""

import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger('artifice')


def plot_image(*images, columns=10, ticks=False):
  columns = min(columns, len(images))
  rows = max(1, len(images) // columns)
  fig, axes = plt.subplots(rows,columns, squeeze=False,
                           figsize=(10, 10*rows/columns))
  for i, image in enumerate(images):
    ax = axes[i // columns, i % columns]
    im = ax.imshow(np.squeeze(image), cmap='gray', vmin=0., vmax=1.)
  if not ticks:
    for ax in axes.ravel():
      ax.axis('off')
      ax.set_aspect('equal')
  fig.subplots_adjust(wspace=0, hspace=0)
  return fig, axes

def plot_detection(label, detection, *images):
  fig, axes = plot_image(*images)
  axes = np.squeeze(axes)
  for i in range(axes.shape[0]):
    axes[i].plot(label[:,2], label[:,1], 'g+', markersize=8., label='known position')
    axes[i].plot(detection[:,2], detection[:,1], 'rx', markersize=8.,
                 label='model prediction')
  axes[-1].legend(loc='upper right')
  fig.suptitle('Object Detection')
  return fig, axes
