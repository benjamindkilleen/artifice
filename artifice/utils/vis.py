"""Utils for visualizing artifice output. (Mostly for testing).

TODO: make `show` functions wrappers around `plot` functions, which can be
called without clearing the matplotlib buffer.

"""

import matplotlib.pyplot as plt
import numpy as np

def show_predict(image, annotation, prediction):
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
  plt.show()


def show_labels(labels, bins=50):
  fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)
  axes[0].hist(labels[:,0,1], bins)
  axes[0].set_title("Sphere 1 X Positions", pad=-25)
  
  axes[1].hist(labels[:,0,2], bins)
  axes[1].set_title("Sphere 1 Y Positions", pad=-25)
  
  axes[2].hist(labels[:,1,1], bins)
  axes[2].set_title("Sphere 2 X Positions", pad=-25)
  
  axes[3].hist(labels[:,1,2], bins)
  axes[3].set_title("Sphere 2 Y Positions", pad=-25)
  plt.show()

def show_scene(scene):
  """Shows a scene's image, annotation, and label.

  :param scene: 
  :returns: 
  :rtype: 

  """
  image, (annotation, label) = scene

  fig, axes = plt.subplots(1, 3)
  axes[0].imshow(np.squeeze(image), cmap='gray')
  axes[1].imshow(np.squeeze(annotation))
  axes[3].text(str(label))
  plt.show()

def show_background(background):
  plt.imshow(np.squeeze(background), cmap='gray')
  plt.colorbar(orientation='vertical')
  plt.show()

