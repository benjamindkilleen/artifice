"""Find new points in label space which would enrich or "smooth" the dataset
according to some distribution.

Brainstorm:
So the labels come in, and we just need to essentially reduce the ks statistic
until the density reaches a certain level? Or until the ks statistic is below a
given threshold. That should do it.

So each 1-D Smoother keeps track of which points were added and which points are
original. When it's done, the 3-D smoother reassembles novel points by combining
all possible examples from the new label points.

"""

import numpy as np
import itertools
import logging

logger = logging.getLogger('artifice')

class Smoother(object):
  """Implement additive uniform smoothing on each dimension of a sample,
  minimizing the Kolmogorov-Smirnov statistic for a given distribution.
  :math:`\sup_y |F_n(y) - F(y)|`, where :math:`F` is the desired comulative
  distribution and :math:`F_n` is the empirical distribution of the sample.
  
  maintains a sorted array of all `sample` points as well as an unsorted
  `inserted` that maintains the non-original values.

  The `smooth()` method is used to add new samples until the 

  """

  def __init__(self, sample, **kwargs):
    """

    :param sample: a 1-D numpy array of points.
    :param lower: lower bound of the sample
    :param upper: upper bound of the sample
    :returns: 
    :rtype: 

    """
    assert sample.ndim == 1
    self.original = sample
    self.sample = np.sort(sample)
    self._lower = kwargs.get('lower')
    self._upper = kwargs.get('upper')
    self._inserted = []
    
    if self.lower > self.sample[0]:
      logger.warning(f"Eliminating points below {self.lower} lower bound.")
      self.sample = self.sample[self.lower <= self.sample] # to keep
    if self.upper < self.sample[-1]:
      logger.warning(f"Eliminating points above {self.upper} upper bound.")
      self.sample = self.sample[self.upper >= self.sample]  
    
  @property
  def inserted(self):
    return np.array(self._inserted)
    
  @property
  def lower(self):
    # TODO: use -inf as the default lower bound
    if self._lower is None:
      return self.sample[0]
    else:
      return self._lower

  def set_lower(self, lower):
    self._lower = lower

  @property
  def upper(self):
    # TODO: use inf as the default upper bound
    if self._upper is None:
      return self.sample[-1]
    else:
      return self._upper

  def set_upper(self, upper):
    self._upper = upper

  @property
  def n(self):
    return self.sample.shape[0]

  def __getitem__(self, i):
    """Include self.lower and self.upper in (simple) indexing."""
    if i < -1:
      raise ValueError
    elif i == -1:
      return self.lower
    elif i == self.n:
      return self.upper
    else:
      return self.sample[i]

  def interval_at(self, i):
    """Return the sample interval right of the i'th data point.

    Robust to multiple points being at index i. If there is no data 

    :param i: index of the point left of the interval. May be -1 to indicate the
    lower boundary.
    :returns: interval
    :rtype: 

    """
    if i >= self.n:
      raise RuntimeError(f"Cannot get item {i} of sample with {self.n} items.")

    left = self[i]
    j = i + 1
    while j < self.n and self[j] == left:
      j += 1
    right = self[j]
        
    return left,right

  def insert(self, y, adjust_bounds=False):
    """Insert new sample values y.

    :param y: value or array of new values to insert.
    :param adjust_bounds: keep new values outside `[self.lower, self.upper]` and
    adjust accordingly. False (default) discards these values.
    :returns: self

    """
    y = np.array(y).reshape(-1)
    if adjust_bounds:
      if np.any(y < self.lower):
        self._lower = np.min(y)
      if np.any(y > self.upper):
        self._upper = np.max(y)
      values = y
    else:
      values = y[y >= self.lower and y <= self.upper]
      if values.shape[0] == 0:
        return self.sample
    
    indices = np.searchsorted(self.sample, y)
    self.sample = np.insert(self.sample, indices, y)
    self._inserted += y.tolist()
    return self

  def cdf(self, y):
    """Cumulative distribution function for :math:`F`.

    By default this is a uniform distribution over `[self.lower, self.upper]`,
    but subclasses may implement their own distribution.

    :param y: where to calculate the distribution
    :returns: F(y)
    :rtype: 

    """
    return (y - self.lower) / (self.upper - self.lower)

  def draw(self, lo, hi):
    """Draw a random value to insert in the interval `[lo,hi]`.

    By default, draws from a uniform distribution, but subclasses should
    implement their own distribution.

    TODO: use scipy.stats.rv_continuous to draw a better random value. Uniform
    is fine if F is a uniform distribution, or in the limit, but in general this
    distribution should match the underlying desired one.

    :param lo: left endpoint
    :param hi: right endpoint
    :returns: random value in `[lo,hi]`
    :rtype: float

    """
    return np.random.uniform(lo, hi)

  def empirical(self, y, strict=False):
    """Compute empirical distribution of the sample :math:`F_n(y)`.

    :param y: single point or array of poitns
    :param strict: use a strict comparison. When y is equal to a sample point,
    this computes :math:`\lim_{y' \rightarrow y} F_n(y')` from the left.
    :returns: 

    """
    ys = np.array(y).reshape(-1, 1)    # [[y_0],[y_1],..., [y_m-1]], (m,1)
    sample = self.sample[np.newaxis,:] # [[s_0, s_1,..., s_n-1]], (1,n)

    # [[s_0,s_1,... >= y_0], [s_0,s_1,... >= y_1], ...], (m,n)
    if strict:
      indicators = np.less(sample, ys)
    else:
      indicators = np.less_equal(sample, ys)

    return np.sum(indicators, axis=1) / self.n

  def ks_diff(self, y, strict=False):
    """Compute the difference :math:`|F_n(y) - F(y)|`.

    :param y: point or array of points
    :param strict: use a strict comparison. When y is equal to a sample point,
    this computes :math:`\lim_{y' \rightarrow y} |F_n(y) - F(y)|` from the
    left.
    :returns: difference between empirical and theoretical distributions at y

    """

    return np.absolute(self.empirical(y) - self.cdf(y))

  def _ks_info(self):
    """Find the interval and value at which :math:`|F_n(y) - F(y)|` has its
    supremum.

    Calculate the statistic from among the points in the sample. In the case
    where F_n(y) >= F(y), the greatest difference in each interval is at its left
    endpoint (because F is nondecreasing and F_n is constant in the
    interval). In the case where F_n <= F(y), the greatest difference is at the
    right endpoint, taking the limit in the case of F_n. There is a third case
    where F crosses F_n in the interval, but it works out similarly.

    :returns diff: the value of the supremum
    :returns interval: tuple `(l,r)` the interval in which diff was found.
    :returns differences: for every interval, TODO: necessary?
    :rtype: tuple of type (float, (float, float))

    """
    left_side_diffs = np.empty(self.n + 1)
    left_side_diffs[0] = self.difference_at(self.lower)
    left_side_diffs[1:] = self.difference_at(self.sample)

    right_side_diffs = np.empty(self.n + 1)
    right_side_diffs[:-1] = self.difference_at(self.sample, strict=True)
    right_side_diffs[-1] = self.difference_at(self.upper, strict=True)

    # Find the interval
    left_idx = np.argmax(left_side_diffs) - 1 # so that self.lower has idx -1
    right_idx = np.argmax(right_side_diffs)
    left_side_diff = left_side_diffs[left_idx]
    right_side_diff = right_side_diffs[right_idx]

    if left_side_diff > right_side_diff:
      diff = left_side_diff
      interval = self.interval_at(left_idx)
    else:
      diff = right_side_diff
      interval = self.interval_at(right_idx - 1)

    differences = np.where(left_side_diffs > right_side_diffs,
                           left_side_diffs, right_side_diffs)

    return diff, interval, differences

  def ks(self):
    """Find the KS statistic :math:`\sup_y |F_n(y) - F(y)|`

    Wrapper around _ks_info that returns only the difference. Should also
    be used for the __call__ method.

    :returns: :math:`\sup_y |F_n(y) - F(y)|`
    :rtype: float

    """
    return self._ks_info()[0]
  
  def _ks_interval(self):
    """Find the interval in which :math:`\sup_y |F_n(y) - F(y)|` is maximized.

    :returns: endpoints of the interval
    :rtype: tuple of floats

    """
    return self._ks_info()[1]
  
  def new_point_interval(self):
    """Find the interval in which a new point should be inserted.

    Get the supremum of the KS statistic in each interval of the sample (at the
    limit of either endpoint), and return the maximum interval after weighting
    by width (prioritizing empty parts of the distribution).

    TODO: reconsider this weighting strategy. If the lower and upper bounds are
    very far away, then those intervals will always be prioritized as being the
    largest? Weighting by width converges MUCH faster though. Unweighted
    adjustments don't seem to converge at all.

    """
    _, _, differences = self._ks_info()

    weighted_diffs = np.empty_like(differences)
    for i in range(differences.shape[0]):
      width = self[i] - self[i-1]
      weighted_diffs[i] = width * differences[i]
    idx = np.argmax(weighted_diffs) - 1 # accounts for lower bound interval
    return self.interval_at(idx)
  
  def new_point(self):
    """Return a new point to insert, performing smoothing."""
    return self.draw(*self.new_point_interval())

  def insert_new_point(self):
    """Insert a single new example according to additive smoothing.

    :returns: self

    """
    return self.insert(self.new_point())
  
  def smooth(self, threshold=0.01, max_iter=None):
    """Smooth the sample until its ks statistic is below `threshold`

    :param threshold: Default is 0.01
    :param max_iter: iteration limit. Default (None) adds points forever until
    threshold reached.
    :returns: self
    :rtype: 

    """
    if max_iter is None:
      steps = itertools.count()
    else:
      steps = range(max_iter)
      
    for i in steps:
      self.insert_new_point()
      ks = self.ks(recent=True)
      if i % 10 == 0:
        logging.info(f"added {i}/{num_iter}, ks = {ks}")
      if ks <= threshold:
        break
    
    return self.ks(recent=True)

  def __call__(self, **kwargs):
    return self.smooth(**kwargs)
  
  
class MultiSmoother(object):
  """A MultiSmoother creates a smoother for every dimension if an N-D sample. It
  is relatively efficient when returning the inserted labels, choosing to return
  an iterable over the cartesian product (in no particular order) rather than
  every possible combination in one array.

  Keep a list for every dimension of the sample. If no smoothing desired for
  that smaple, then entry is None.

  """
  def __init__(self, sample, bounds):
    """Initialize the MultiSmoother.

    :param sample: (num_points, num_dimensions) shaped array of samples from any
    number of distributions. If more than 2D, points are internally flattened
    and then the iterator will reshape them, so that the user can structure
    multi-D samples however they want.
    :param bounds: list of tuples (lower, upper) or None, if no smoothing. Note
    that this must be one-dimensional, corresponding to the flattened points in
    each element of sample.
    desired, for each dimension.

    """
    self._elem_shape = sample.shape[1:] # Used to unflatten in the iterator.
    _sample = sample.reshape(sample.shape[0], -1)
    
    self._bounds = bounds
    self._smoothers = []
    for i, region in enumerate(bounds):
      smoother = (Smoother(_sample[:,i]) if bounds is None else
                  Smoother(_sample[:,i], lower=bounds[0], upper=bounds[1]))
      self._smoothers.append(smoother)

  def smooth(self, threshold=0.01, max_iter=None):
    """Smooth each dimension in the sample.

    TODO: allow for threshold and max_iter to be specified for each dimension.

    :param threshold: ks threshold
    :param max_iter: maximum iterations
    :returns: 
    :rtype: 

    """
    
    for smoother, bound in zip(self._smoothers, self._bounds):
      if bound is None:
        continue
      smoother.smooth(threshold, max_iter)

  @property
  def inserted(self):
    """Return an iterable over all the inserted points.

    This is a cartesian product over all points in every smoothed sample,
    ignoring the originals.

    For the completely novel points, this is easy. Just use
    itertools.product or something like it. But also need to include the
    elements from originals, just not a point with all originals.

    Can iterate over the dimensions, and at each dimension, take the product of
    that dimension's inserted points and every other dimension's full
    sample. TODO: make sure no repetitions.

    So at the first dimension, take all the inserted points there, cartesian
    product with the full sample of the rest.
    
    Considering just inserted points at the second dimension, the only points we
    haven't seen are ones where the first dimension is constrained to original points.

    """
    iterators = []
    for d, smoother in enumerate(self._smoothers):
      points = (
        [self._smoothers[i].original for i in range(d)]
        + [self._smoothers[d].inserted]
        + [self._smoothers[i].sample for i in range(d+1, len(self._smoothers))])
      iterator = map(lambda t : np.array(t).reshape(*self._elem_shape),
                     itertools.product(*points))
      iterators.append(iterator)

    return itertools.chain(iterators)
