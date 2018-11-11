"""Implements artifice's detection scheme from end to end.

Training scheme:
* load a dataset, which may or may not be labeled with full annotations.
* For each batch, obtain the semantic annotations from labeller.py. (This may
  involve just taking the first num_classes channels of full_annotation.)
* Perform any augumentations using that newly-labelled data. Add all of this to
  the dataset, which should be rewritten? Unclear how to keep track of a
  constantly changing dataset.
* Train the semantic segmentation on this batch, using semantic_model.py.
* 

"""

import tensorflow as tf

from artifice.semantic_segmentation import UNet
from artifice.labeller import Labeller

"""Artifice's full detection scheme.
"""
class Model:
  def __init__(self, image_shape, num_classes, sem_model=UNet):
    self.semantic_model = sem_model(image_shape, num_classes)
    

  def train(self, data):
    """Implement the full training scheme for artifice. Unlike most "train"
    functions for a tensorflow, this does not perform a test-train
    split. 

    In user mode, this will 
    
    """

    # load the dataset, with or without full annotations?

    # obtain semantic annotations
    
    
