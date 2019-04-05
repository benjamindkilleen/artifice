import tensorflow as tf
from artifice import mod, dat
import logging

logger = logging.getLogger('artifice')

class ActiveLearner():
  def __init__(self, model):
    """Wrapper around model's that performs active learning on dat.Data objects.

    :param model: a `mod.Model`
    :returns: 
    :rtype: 

    """
    self.model = model
    self.annotated_set = None
    
  def fit(self, unlabeled_set):
    """

    :param unlabeled_set: a dat.Data object with an `annotate()` method.
    :returns: 
    :rtype: 

    """
    
    sampling = np.zeros(unlabeled_set.size, np.int64)
    sampling[0] = 1

    
