"""Test the ability to sample from a distribution additively.

"""

import numpy as np
import matplotlib.pyplot as plt
from artifice.utils import video
import logging
from sys import argv

logging.basicConfig(level=logging.INFO)

class DistributionComparator:
  """Implement a Kolmogorov-Smirnov test on a 1-D sample.

  """
  def __init__(self, sample, **kwargs):
    """Implement a Kolmogorov-Smirnov test.

    :param sample: a SORTED 1D sample.

    """
    assert sample.ndim == 1
    self.sample = np.sort(sample)
    self._lower = kwargs.get('lower')
    self._upper = kwargs.get('upper')

  def distribution(self, y):
    """A vectorized function on y.

    :param y: a point or array of points in data space.
    :returns: 
    :rtype: 

    """
    raise NotImplementedError("Subclasses should implement the distribution.")
  
  @property
  def lower(self):
    if self._lower is None:
      return self.sample[0]
    else:
      return self._lower

  def set_lower(self, lower):
    self._lower = lower

  @property
  def upper(self):
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
  
  def difference_at(self, y, strict=False, squeeze=True):
    """Compute the difference :math:`|F_n(y) - F(y)|`.

    :param y: a single point or array of points
    :param strict: use a strict comparison. When y is equal to a sample point,
    this computes :math:`\lim_{y' \rightarrow y} |F_n(y) - F(y)|` from the
    left.
    :param squeeze: squeeze the dimensions of the output
    :returns: comparison or array of comparisons
    :rtype: 

    """
    ys = np.array(y).reshape(-1, 1)    # [[y_0],[y_1],..., [y_m-1]], (m,1)
    sample = self.sample[np.newaxis,:] # [[s_0, s_1,..., s_n-1]], (1,n)

    # [[s_0,s_1,... >= y_0], [s_0,s_1,... >= y_1], ...], (m,n)
    if strict:
      indicators = np.less(sample, ys)
    else:
      indicators = np.less_equal(sample, ys)

    difference = np.sum(indicators, axis=1) / self.n - self.distribution(y)
    if squeeze:
      difference = np.squeeze(difference)

    return np.absolute(difference)

  def _ks_info(self):
    """Find the interval in which :math:`|F_n(y) - F(y)|` has its supremum.

    Calculate the statistic from among the points in the sample. In the case
    where F_n(y) >= F(y), the greatest difference in each interval is at its left
    endpoint (because F is nondecreasing and F_n is constant in the
    interval). In the case where F_n <= F(y), the greatest difference is at the
    right endpoint, taking the limit in the case of F_n. There is a third case
    where F crosses F_n in the interval, but it works out similarly.

    :returns: (diff, (l,r), diffs), where `diff` is the value of the supremum and
    (l,r) is the interval in which it was found. `differences` is for each interval
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
      l = self[left_idx]
      r = self[left_idx + 1]
    else:
      diff = right_side_diff
      l = self[right_idx - 1]
      r = self[right_idx]

    differences = np.where(left_side_diffs > right_side_diffs,
                           left_side_diffs, right_side_diffs)

    return diff, (l,r), differences
    
  def ks(self):
    """Find the KS statistic :math:`\sup_y |F_n(y) - F(y)|`

    Wrapper around _ks_info that returns only the difference. Should also
    be used for the __call__ method.

    :returns: :math:`\sup_y |F_n(y) - F(y)|`
    :rtype: float

    """
    return self._ks_info()[0]

  def ks_interval(self):
    """Find the interval in which :math:`\sup_y |F_n(y) - F(y)|` is maximized.

    :returns: endpoints of the interval
    :rtype: tuple of floats

    """
    return self._ks_info()[1]

  def __call__(self):
    return self.ks()

  def draw_point_in(self, lo, hi):
    """Generate a new point to insert in the interval.

    By default, draw from a uniform distribution between lo and hi, but
    subclasses can implement their own distribution.

    :param lo: range lower boundary
    :param hi: range upper boundary
    :returns: new point value
    :rtype: float

    """
    # TODO: determine a better distribution? Requires further analysis.
    return np.random.uniform(lo, hi)
    
  def new_point_interval(self):
    """Find the interval in which a new point should be inserted.

    Get the suprumum of the statistic in each interval, return the maximum after
    weighting by each interval width.
    
    :returns: lo, hi of the range
    :rtype: tuple

    """

    _, _, differences = self._ks_info()

    weighted_diffs = np.empty_like(differences)
    for i in range(differences.shape[0]):
      width = self[i] - self[i-1]
      weighted_diffs[i] = width * differences[i]

    idx = np.argmax(weighted_diffs) - 1
    return self.interval_at(idx)

  
  def new_point(self):
    """Return a new point to insert, performing smoothing."""
    return self.draw_point_in(*self.new_point_interval())

  def insert_new_point(self):
    """Insert a single new example according to additive matching.

    :returns: self

    """
    return self.insert(self.new_point())

  def insert(self, y, adjust_bounds=False):
    """Insert new sample values y.

    :param y: value or array of new values to insert.
    :param adjust_bounds: keep new values outside `[self.lower, self.upper]` and
    adjust accordingly. False (default) discards these values.
    :returns: self
    :rtype: 

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
        raise Warning("All values outside [lower, uppper].")
        return self.sample
    
    indices = np.searchsorted(self.sample, y)
    self.sample = np.insert(self.sample, indices, y)
    return self
  
  def plot_sample(self):
    """Plot the sample histogram.

    Uses a constant number of bins and other settings (so that repeated calls
    can be strung together).

    :returns: image array of the plot

    """
    fig, ax = plt.subplots(1,1)
    ax.hist(self.sample, bins = 30, range=(self.lower, self.upper),
            density=False)
    ax.set_title('Sample Histogram')
    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image
  
    
class UniformComparator(DistributionComparator):
  def distribution(self, y):
    return (y - self.lower) / (self.upper - self.lower)
  

def main():
  """Show the smoothing for a sharply spiked gaussian.

  :returns: 
  :rtype: 

  """
  num_iter = 1 if len(argv) == 1 else int(argv[1])
  n = 100
  lower = -5.
  upper = 5.
  sample = np.clip(np.random.normal(size=n), lower, upper)
  comparator = UniformComparator(sample, lower=lower, upper=upper)
  frame = comparator.plot_sample()
  
  writer = video.MP4Writer('docs/additive_smoothing.mp4', frame.shape, fps=50)
  writer.write(frame)
  for i in range(num_iter):
    if i % 10 == 0:
      logging.info(f"adding {i}/{num_iter}")
    comparator.insert_new_point()
    frame = comparator.plot_sample()
    writer.write(frame)
  
  writer.close()

if __name__ == '__main__':
  main()
