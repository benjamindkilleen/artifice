"""Module for artifice's selector.

"""

from time import sleep, time
import numpy as np
import tensorflow as tf

from artifice.log import logger
from artifice import ann

class Prioritizer:
  def __init__(self, data_set, *, info_path):
    self.data_set = data_set
    self.info = ann.AnnotationInfo(info_path, clear_priorities=True,
                                   clear_limbo=False)

  def run(self, seconds=-1):
    """Run for at most `seconds`. If `seconds` is negative, run forever."""
    start_time = time()
    if tf.executing_eagerly():
      for indices, images in self.data_set.enumerated_prediction_input().repeat(-1):
        logger.info(f"evaluating priorities for {indices}...")
        priorities = list(self.prioritize(images))
        self.info.push(list(zip(list(indices), priorities)))
        logger.info(f"pushed {indices} with priorities {priorities}.")
        if time() - start_time > seconds > 0:
          logger.info(f"finished after {seconds}s.")
          break
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

class ModelUncertaintyPrioritizer(Prioritizer):
  """Uses the `uncertainty_on_batch` method of ArtificeModel to prioritize each
  image."""
  
  def __init__(self, *args, model, load_freq=200, **kwargs):
    """
    :param model: model to use
    :param load_freq: how frequently to load the weights
    :returns: 
    :rtype: 

    """
    self.model = model
    self.load_freq = load_freq
    self.count = 0
    super().__init__(*args, **kwargs)

  def prioritize(self, images):
    if self.count % self.load_freq == 0:
      self.model.load_weights()
    self.count += 1
    return self.model.uncertainty_on_batch(images)
  
