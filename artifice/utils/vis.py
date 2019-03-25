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
                           figsize=(8*columns/2, 8*rows/2))
  for i, image in enumerate(images):
    ax = axes[i // columns, i % columns]
    im = ax.imshow(np.squeeze(image), cmap='gray', vmin=0., vmax=1.)
  if not ticks:
    for ax in axes.ravel():
      ax.axis('off')
      ax.set_aspect('equal')
  fig.subplots_adjust(wspace=0, hspace=0)

def plot_detection(image, label, detection):
  plot_image(image)
  plt.plot(label[:,2], label[:,1], 'g+', markersize=20., label='Truth')
  plt.plot(detection[:,2], detection[:,1], 'rx', markersize=10.,
           label='Prediction')
  plt.title("Object Detection")
  


