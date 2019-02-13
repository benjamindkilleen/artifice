"""Find new points in label space which would enrich or "smooth" the dataset
according to some distribution.

"""

import numpy as np
import logging

logger = logging.getLogger('artifice')

class Smoother():
  """Implement additive uniform smoothing on each dimension of a sample,
  minimizing the Kolmogorov-Smirnov statistic for a given distribution.

  """

  def __init__(self, sample):
    """FIXME! briefly describe function

    :param sample: a 2-D numpy array.
    :returns: 
    :rtype: 

    """
    
    self.sample = np.sort()
