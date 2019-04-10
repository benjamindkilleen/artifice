"""Messing around with annotating the gyros data. These annotators are highly
ad-hoc and not what would eventually be used, in all likelihood. They're more
the sort of thing we want to avoid."""

import numpy as np
from artifice.utils import img, vis
from glob import glob
from skimage.feature import canny
import matplotlib.pyplot as plt

class Annotator():
  sigma = 1.0
  def annotate_object(self, image, obj_label):
    """Annotate the object at index-space position.

    :param image: numpy image, float32
    :param obj_label: 1D object label with index-space position at obj_label[1:3]
    :returns: `(xs,ys)` lists of indices belonging to the object at position

    """    
    raise NotImplementedError

  def annotate_image(self, image, label):
    """Annotate all the objects in the image.

    :param image: the numpy image
    :param label: `(num_objects, label_dim)` where `label_dim >= 3`
    :returns: 
    :rtype: 

    """
    annotation = np.zeros_like(image)
    grad = np.gradient(image)
    edges = canny(image, sigma=self.sigma)
    for obj_label in label:
      xs, ys = self.annotate_object(image, obj_label)
      annotation[xs,ys] = obj_label[0]
    return annotation

  def __call__(self, *args, **kwargs):
    return self.annotate_image(*args, **kwargs)

class GyroAnnotator(Annotator):
  sigma = 1.7
  def annotate_object(self, image, obj_label, edges=None):
    """Annotate the gyro at `obj_label`.

    Run a canny edge detector on the image, if needed

    Exploits the fact that the gyros should be roughly circular, but the given
    location may not be the center, need to calculate the edge. So, within the
    region. Calculates the
    radius along 8 directions and takes the median of these (not the average, in
    case one example throws everything off). The "radius" is determined to be
    when the image gradient is at a peak.

    """
    if grad is None:
      grad = np.gradient(image)

    
    
    raise NotImplementedError
    
  

def main():
  labels = np.load('data/gyros/original_labels.npy')
  image_paths = sorted(glob('data/gyros/images/*.png'))
  annotator = GyroAnnotator()
  for image_path in image_paths:
    image = img.open_as_float(image_path)
    edges = canny(image, sigma=1.7)
    vis.plot_image(image, edges, scale=50)
    # plt.show()
    plt.savefig('docs/gyros_edges.png')
    break

if __name__ == "__main__":
  main()
