"""Module for artifice's selector.

"""

from time import sleep
import numpy as np
import logging

from artifice import ann

logger = logging.getLogger('artifice')

class Prioritizer:
  def __init__(self, data_set, *, info_path):
    self.data_set = data_set
    self.info = ann.AnnotationInfo(info_path)

  def run(self):
    if tf.executing_eagerly():
      for indices, images in self.data_set.enumerated_prediction_input:
        logger.info(f"evaluating priorities for {indices}...")
        priorities = list(self.prioritize(images))
        self.info.push(list(zip(list(indices), priorities)))
        logger.info(f"pushed {indices} with priorities {priorities}.")
    else:
      raise NotImplementedError("patient execution")

  def prioritize(self, images):
    """Assign a priority to a batch of images for labeling.

    The meaning of "priority" is flexible. In the context of active learning, it
    could be the measure of uncertainty. Higher priority examples will be
    annotated first.

    :param image: batch of images
    :returns: batch of priorities, in numpy or list form
    :rtype:

    """
    raise NotImplementedError("subclasses should implement")

class SimulatedPrioritizer(Prioritizer):
  def __init__(self, *args, selection_delay=1, **kwargs):
    self.selection_delay = selection_delay
    super().__init__(*args, **kwargs)
  
class RandomPrioritizer(SimulatedPrioritizer):
  def prioritize(self, images):
    sleep(self.selection_delay)
    return np.random.uniform(0, 1, size=images.shape[0])
  
prioritizers = {0 : RandomPrioritizer}
