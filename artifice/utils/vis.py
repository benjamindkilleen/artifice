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
                           figsize=(columns/2, rows/2))
  for i, image in enumerate(images):
    ax = axes[i // columns, i % columns]
    im = ax.imshow(np.squeeze(image), cmap='gray', vmin=0., vmax=1.)
  if not ticks:
    for ax in axes.ravel():
      ax.axis('off')
      ax.set_aspect('equal')
  fig.subplots_adjust(wspace=0, hspace=0)  

  
def plot_predict(image, annotation, prediction):
  """Show the output of the model. Meant for testing."""
  fig, axes = plt.subplots(3,2)
  axes[0,0].imshow(np.squeeze(image), cmap='gray')
  axes[0,0].set_title("Original Image")
  axes[0,1].axis('off')
  
  im = axes[1,0].imshow(prediction['logits'][:,:,0], cmap='magma')
  axes[1,0].set_title("Id=0")
  fig.colorbar(im, ax=axes[1,0], orientation='vertical')
  im = axes[1,1].imshow(prediction['logits'][:,:,1], cmap='magma')
  axes[1,1].set_title("Id=1")
  fig.colorbar(im, ax=axes[1,1], orientation='vertical')
  
  axes[2,0].imshow(np.squeeze(annotation))
  axes[2,0].set_title("Annotation")
  axes[2,1].imshow(np.squeeze(prediction['annotation']))
  axes[2,1].set_title("Predicted Annotation")

  
def plot_labels(labels, bins=97):
  _range = [[0, 387], [0, 387]]
  fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
  axes[0].hist2d(labels[:,0,1], labels[:,0,2], bins, range=_range, cmap='magma')
  axes[0].set_title("Sphere 1 Positions")

  axes[1].hist2d(labels[:,1,1], labels[:,1,2], bins, range=_range, cmap='magma')
  axes[1].set_title("Sphere 2 Positions")
  plt.axis('off')

  
def plot_scene(*scenes):
  """Shows a scene's (or multiple) image, annotation, and label.

  :param scene: 
  :returns: 
  :rtype: 

  """

  fig, axes = plt.subplots(len(scenes), 3, squeeze=False)
  for i, scene in enumerate(scenes):
    image, (annotation, label) = scene
    axes[i,0].imshow(np.squeeze(image), cmap='gray')

    im = axes[i,1].imshow(annotation[:,:,0])
    axes[i,1].set_title("Semantic Annotation")

    im = axes[i,2].imshow(np.clip(annotation[:,:,1], 0, 30), cmap='magma')
    axes[i,2].set_title("Distance Annotation")
    fig.colorbar(im, ax=axes[i,2], orientation='vertical')


def plot_background(background):
  plt.figure()
  plt.imshow(np.squeeze(background), cmap='gray')
  plt.colorbar(orientation='vertical')

