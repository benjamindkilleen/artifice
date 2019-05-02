"""Messing around with annotating the gyros data. These annotators are highly
ad-hoc and not what would eventually be used, in all likelihood. They're more
the sort of thing we want to avoid."""

import numpy as np
from artifice.utils import img, vis
from glob import glob
from skimage.feature import canny
from skimage.draw import circle
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
    edges = canny(image, sigma=self.sigma)
    for obj_label in label:
      xs, ys = self.annotate_object(image, obj_label, edges=edges)
      annotation[xs,ys] = obj_label[0]
    return annotation

  def __call__(self, *args, **kwargs):
    return self.annotate_image(*args, **kwargs)

class GyroAnnotator(Annotator):
  initial_radius = 10
  def annotate_object(self, image, obj_label, edges=None):
    """Annotate the gyro at `obj_label`.

    Run a canny edge detector on the image, if needed

    """
    if edges is None:
      edges = canny(image, sigma=self.sigma)

    """
    Grab the edge pixels within 10 pixels, calculate distance to center, take
    their median as the true radius and return pixels within that range.
    """

    rr, cc = circle(obj_label[1], obj_label[2], self.initial_radius,
                    shape=image.shape)

    mask = np.zeros_like(image, dtype=bool)
    mask[rr,cc] = True
    xs, ys = np.where(np.logical_and(edges, mask))
    distances = np.linalg.norm(
      np.stack((xs - obj_label[1], ys - obj_label[2]), axis=1), axis=1)
    r = np.median(distances) + 1.5
    return circle(obj_label[1], obj_label[2], r, shape=image.shape)
  
def main():
  labels = np.load('data/gyros/labels.npy')
  image_paths = sorted(glob('data/gyros/images/*.png'))
  annotator = GyroAnnotator()
  for i, (image_path, label) in enumerate(zip(image_paths, labels)):
    if i % 100 == 0:
      print(f"{i} / {labels.shape[0]}")
    image = img.open_as_float(image_path)
    annotation = annotator(image, labels[0])
    np.save(f'data/gyros/annotations/{str(i).zfill(4)}.npy', annotation)
    # fig, axes = vis.plot_image(image, scale=80)
    # xs, ys = np.where(annotation)
    # plt.plot(ys, xs, 'r.')
    # plt.show()
    # plt.savefig('docs/gyro_annotation.png')

if __name__ == "__main__":
  main()
