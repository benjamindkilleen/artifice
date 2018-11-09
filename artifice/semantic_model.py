"""Provides semantic segmentation capability for artifice. The specific
algorithm isn't important, as long as it returns an output in familiar form.

We implement the U-Net segmentation architecture
(http://arxiv.org/abs/1505.04597), emulating implementations at
* https://github.com/jakeret/tf_unet/
* https://github.com/tks10/segmentation_unet/

"""

import os
import numpy as np
import logging

import tensorflow as tf
from artifice.utils import dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class UNet(object):
  """Implementation of UNet.
  args:
    channels: number of channels in the image (grayscale by default)
    num_classes: number of classes of objects to be detected, not including
      background class 
    kwargs: passed to create_unet_graph
  """

  def __init__(self, channels=1, num_classes=1, **kwargs):
    self.channels = channels
    self.n_class = num_classes + 1 # include background class

    self.x = tf.placeholder("float", shape=[None, None, None, self.channels], name='x')
    self.y = tf.placeholder("float", shape=[None, None, None, self.n_class], name='y')
    self.keep_probability = tf.placeholder(tf.float32, name="dropout_probability")
    
    # TODO: self
    logits, self.weights, self.offset = self._create_graph(**kwargs)

  
  def _create_graph(self, layers=3, features_root=16, filter_size=3, pool_size=2,
                    summaries=True):
    """Create the unet graph. Should only be called by init, after self.x,
    self.channels, and self.n_class have been set. """
    pass
