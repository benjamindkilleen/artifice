"""Module for artifice's selector.

"""

import numpy as np
from artifice import ann

class Selector:
  def __init__(self, data_set, *, info_path):
    self.data_set = data_set
    self.info = ann.AnnotationInfo(info_path)

  def __call__(self):
    while True:
      pass
    # todo: implement this, while true call prioritize on images and add them to
    # the queue. Should be able to use the batch size of a model reasonably.

  def prioritize(self, image):
    """Assign a priority to an image for labeling.

    The meaning of "priority" is flexible. In the context of active learning, it
    could be the measure of uncertainty. Higher priority examples will be
    annotated first.

    :param image: image or batch of images
    :returns: priority or batch of priorities
    :rtype: 

    """
    raise NotImplementedError("subclasses should implement")

class RandomSelector:
  def prioritize(self, _):
    return float(np.random.uniform(0, 1))
  
selectors = {0 : RandomSelector}
