"""Utils for visualizing artifice output. (Mostly for testing).

TODO: make `show` functions wrappers around `plot` functions, which can be
called without clearing the matplotlib buffer.

"""

import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger('artifice')


def plot_image(*images, columns=10, ticks=False, scale=20):
  columns = min(columns, len(images))
  rows = max(1, len(images) // columns)
  fig, axes = plt.subplots(rows,columns, squeeze=False,
                           figsize=(scale, scale*rows/columns))
  for i, image in enumerate(images):
    ax = axes[i // columns, i % columns]
    im = ax.imshow(np.squeeze(image), cmap='gray', vmin=0., vmax=1.)
  if not ticks:
    for ax in axes.ravel():
      ax.axis('off')
      ax.set_aspect('equal')
  fig.subplots_adjust(wspace=0, hspace=0)
  return fig, axes

def plot_detection(label, detection, *images, n=1):
  """Plot the detections onto the image.

  Allows for multiple images to be plotted, but by default only plots the
  detection onto the first one.

  :param label: 
  :param detection: 
  :param n: how many of `images` to plot onto.
  :returns: 
  :rtype: 

  """
  fig, axes = plot_image(*images)
  for ax in axes.flat[:n]:
    if label is not None:
      ax.plot(label[:,2], label[:,1], 'g+', markersize=8., label='known position')
    if detection is not None:
      ax.plot(detection[:,2], detection[:,1], 'rx', markersize=8.,
                   label='model prediction')
  axes.flat[0].legend(loc='upper left')
  fig.suptitle('Object Detection')
  return fig, axes

def plot_labels(labels, image_shape):
  fig, axes = plt.subplots(1, labels.shape[1])
  for i in range(labels.shape[1]):
    axes[i].hist2d(labels[:,i,2], labels[:,i,1],
                   bins=[image_shape[1],image_shape[0]],
                   range=[[0,image_shape[1]],[0,image_shape[0]]],
                   normed=True)
    axes[i].set_ylim(image_shape[0],0)
    axes[i].axis('off')
    axes[i].set_aspect('equal')
    axes[i].set_title(f"Object '{int(labels[0,i,0])}'")
  fig.suptitle('Object Positions')
