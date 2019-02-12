"""Test the ability to sample from a distribution additively.

"""

import numpy as np
import matplotlib.pyplot as plt
from artifice.utils import video

class DistributionComparator:
  """Implement a Kolmogorov-Smirnov test.

  """
  def __init__(self, sample, **kwargs):
    """Implement a Kolmogorov-Smirnov test.

    :param sample: a SORTED 1D sample.

    """
    self.sample = sample.sort()
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
  
  def difference_at(self, y, signed=False, strict=False, squeeze=True):
    """Compute the difference :math:`|F_n(y) - F(y)|`.

    :math:`F_n` is the empirical distribution of the sample, :math:`F` is the
    distribution.

    :param y: a single point or array of points
    :param signed: return the difference between the empirical and the
    distribution at `y`, without taking the absolute value. Default = False
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
      indicators = np.greater(sample, ys)
    else:
      indicators = np.greater_equal(sample, ys)

    difference = np.sum(indicators, axis=1) / self.n - self.distribution(y)
    if squeeze:
      difference = np.squeeze(difference)

    if signed:
      return difference
    else:
      return np.absolute(difference)

  def ks_statistic(self):
    """Find the KS statistic: sup_y |F_n(y) - F(y)|

    Calculate the statistic from among the points in the sample. In the case
    where F_n(y) >= F(y), the greatest difference in each interval is at its left
    endpoint (because F is nondecreasing and F_n is constant in the
    interval). In the case where F_n <= F(y), the greatest difference is at the
    right endpoint, taking the limit in the case of F_n.

    :returns: sup_y |F_n(y) - F(y)|
    :rtype: float

    """
    left_side_differences = np.empty(self.n + 1)
    right_side_differences = np.empty(self.n + 1)

    left_side_differences[0] = self.difference_at(self.lower)
    left_side_differences[1:] = self.difference_at(self.sample)

    right_side_differences[1:] = self.difference_at(self.sample, strict=True)
    right_side_differences[-1] = self.difference_at(self.upper, strict=True)

    left_idx = np.argmax(left_side_differences)
    right_idx = np.argmax(right_side_differences)
    left_value = self.lower if left_idx == 0 else self.sample[left_idx - 1]
    right_value = (self.upper if right_idx == self.n
                   else self.sample[right_idx])

    return max(left_value, right_value)

  def draw_sample(self, lo, hi):
    """Generate a new sample to insert in the range.

    By default, draw from a uniform distribution between lo and hi, but
    subclasses can implement their own distribution.

    :param lo: range lower boundary
    :param hi: range upper boundary
    :returns: new sample value
    :rtype: float

    """
    return np.random.uniform(lo, hi)
    
  def new_sample_range(self):
    """Find the range in which a new sample should be inserted.

    Iterate over the ranges. For each, get the average of the (signed) KS
    difference at each endpoint, and return the range with the (signed) minimum
    difference.
    
    :returns: lo, hi of the range
    :rtype: tuple

    """
    differences = np.zeros(self.n + 2)
    differences[0] = self.difference_at(self.lower, signed=True)
    differences[1:-1] = self.difference_at(self.sample, signed=True)
    differences[-1] = self.difference_at(self.upper, signed=True)

    range_differences = np.zeros(self.n + 1)
    for i in range_differences.shape[0]:
      range_differences[i] = differences[i] + differences[i+1] / 2

    which_range = np.argmin(range_differences)
    if which_range == 0:
      return self.lower, self.sample[0]
    elif which_range == self.n:
      return self.sample[which_range - 1], self.upper
    else:
      return self.sample[which_range - 1], self.sample[which_range]

  def new_sample(self):
    """Return a new sample to insert, performing smoothing."""
    return self.draw_sample(*self.new_sample_range())

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

  def insert_new_sample(self):
    """Insert a single new example according to additive matching.

    :returns: self

    """
    return self.insert(self.new_sample())

  def plot_sample(self):
    """Plot the sample histogram.

    Uses a constant number of bins and other settings (so that repeated calls
    can be strung together).

    :returns: image array of the plot

    """
    fig, ax = plt.subplots(1,1)
    ax.hist(self.sample, bins = 30, range=(self.lower, self.upper),
            density=True)
    ax.set_title('Sample Histogram')
    fig.canvas.draw()

    plot_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    return plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  
    
class UniformComparator(DistributionComparator):
  def distribution(self, y):
    return (y - self.lower) / (self.upper - self.lower)
  

def main():
  """Show the smoothing for a sharply spiked gaussian.

  :returns: 
  :rtype: 

  """
  n = 100
  lower = -5.
  upper = 5.
  num_iter = 500
  sample = np.clip(np.random.normal(size=n), lower, upper)
  comparator = UniformComparator(sample, lower=lower, upper=upper)
  frame = comparator.plot_sample()
  
  writer = video.MP4Writer('additive_smoothing.mp4', frame.shape, fps=50)
  writer.write(frame)
  for i in num_iter:
    frame = comparator.insert_new_sample().plot_sample()
    writer.write(frame)
    
  writer.close()

if __name__ == '__main__':
  main()
