"""experiments is a tool for creating large, labeled training sets for semantic
segmentation and/or object detection, with the ray-tracing tool POV-Ray.

Dependencies:
* numpy
* POV-Ray
* vapory
* tensorflow

.tfrecord writer largely based on:
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

"""

import numpy as np
import vapory
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import draw

# helper functions for writing tfrecords
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class DynamicObject:
  """An ImageObject is a wrapper around a Vapory object. This may be used for any
  objects in a scene which change from image to image but are not being tracked.
  The vapory object itself is created on every __call__().

  args:
    vapory_object: the vapory class this ExperimentObject represents.
    object_args: either a tuple, containing all required arguments for creating
      vapory_object, or a function which creates this tuple (allowing for
      non-determinism), as well as any others that should change on each call.
    args: any additional args, passed on as vapory_object(*object_args + args)
    kwargs: additions keyword arguments are passed onto the vapory object, the
      same on every call.

  self.args stores the most recent args as a list. This is instantiated as [],
  and is used by Experiment.compute_mask() to generate the mask of the object,
  as well as by Experiment.compute_target() for any additional target values.

  """
  def __init__(self, vapory_object, object_args, *args, **kwargs):
    self.vapory_object = vapory_object
    
    if callable(object_args):
      self.get_args = object_args
    elif type(object_args) == tuple:
      self.get_args = lambda : object_args
    else:
      raise RuntimeError("`object_args` is not a tuple or function")

    self.other_args = list(args) + sum([list(t) for t in kwargs.items()], [])
    self.args = []

  def __call__(self):
    """Return an instance of the represented vapory object, ready to be inserted
    into a scene.
    """
    self.args = list(self.get_args()) + self.other_args
    return self.vapory_object(*self.args)

  
class ExperimentObject(DynamicObject):
  """An ExperimentObject is a wrapper around a Vapory object, and it represents a
  marker for detection. An ExperimentObject should be used, rather than a mere
  ImageObject or vapory object, whenever the object is being tracked (needs a
  mask). Unlike an ImageObject, every ExperimentObject has a class associated
  with it (an integer >0, since 0 is the background class).

  args:
    vapory_object: the vapory class this ExperimentObject represents.
    object_args: either a tuple, containing all required arguments for creating
      vapory_object, or a function which creates this tuple (allowing for
      non-determinism), as well as any others that should change on each call.
    object_class: numerical class of the object, used for generating a
      mask. default=1.
    args: any additional args, passed on as vapory_object(*object_args + args)
    kwargs: additions keyword arguments are passed onto the vapory object, the
      same on every call.

  """

  def __init__(self, vapory_object, object_args, *args, object_class=1, **kwargs):
    super().__init__(vapory_object, object_args, *args, **kwargs)
    assert(object_class > 0)
    self.object_class = int(object_class)

  def compute_mask(experiment):
    """Compute the numpy mask of the ExperimentObject, given Experiment
    `experiment`.

    This should be overwritten by subclasses, for each type of vapory
    object. Each object returns the indices of the experiment scene that it
    contains (from skimage.draw). It is up to the Experiment to decide, in the
    case of occlusions, which object is in front of the other.

    """
    raise NotImplementedError("compute_mask is specific to each vapory object.")

class ExperimentSphere(ExperimentObject):
  """An ExperimentSphere, representing a vapory.Sphere.

  args:
    vapory_object: the vapory class this ExperimentObject represents.
    object_args: either a tuple, containing all required arguments for creating
      vapory_object, or a function which creates this tuple (allowing for
      non-determinism), as well as any others that should change on each call.
    object_class: numerical class of the object, used for generating a
      mask. default=1.
    args: any additional args, passed on as vapory_object(*object_args + args)
    kwargs: additions keyword arguments are passed onto the vapory object, the
      same on every call.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(vapory.Sphere, *args, **kwargs)

  def compute_mask(experiment):
    """Compute the mask for an ExperimentSphere, placed in experiment."""
    assert(len(self.args) != 0)
    
    
class Experiment:
  """An Experiment contains information for generating a dataset, which is done
  using self.run(). It has variations that affect the output labels.

  args:
    img_shape: (rows, cols) shape of the output images, determines the aspect ratio
      of the camera, default=(512,512). Number of channels determined by `mode`
    mode: image mode to generate, default='L' (8-bit grayscale)
    num_classes: number of classes to be detected.
    N: number of images to generate, default=1000
    output_format: filetype to write, default='tfrecord'
    fname: name of output file, without extension. Ignored if included.
    camera_multiplier: controls how far out the camera is positioned, as a
      multiple of img_shape[1] (vertical pixels), default=4 (far away)

  Image `mode` is according to PIL.Image. Valid inputs are:
  * L (8-bit pixels, black and white)
  * RGB (3x8-bit pixels, true colour)
  (other modes to be supported in future versions)

  The camera will be placed in each experiment such that the <x,y,0> plane is
  the image plane, with one unit of distance corresponding to ~1 pixel on that
  plane.

  "targets" are features of the experiment, other than the segmentation mask
  that are ultimately desired, such as position/orientation.

  self.objects is a list of ExperimentObjects that are subject to change,
  whereas self.static_objects is a list of vapory Objects ready to be inserted
  in the scene, as is.
  """

  supported_modes = {'L', 'RGB'}
  supported_formats = {'tfrecord'}
  included = ["colors.inc", "textures.inc"]

  def __init__(self, img_shape=(512,512), mode='RGB', num_classes=1, N=1000,
               output_format='tfrecord', fname="out", camera_multiplier=4):
    assert(type(img_shape) == tuple and len(img_shape) == 2)
    assert(mode in self.supported_modes)
    assert(output_format in self.supported_formats)
    assert(type(fname) == str)
    fname = '.'.join(fname.split('.')[:-1])
    assert(camera_multiplier > 0)
    assert(num_classes > 0)
    
    self.img_shape = img_shape
    self.N = int(N)
    self.mode = mode
    self.output_format=output_format
    self.fname = fname
    self.camera_multiplier = camera_multiplier
    
    self.set_camera()

    # The objects in the scene should be added to by the subclass.
    self.experiment_objects = [] # ExperimentObject instances
    self.dynamic_objects = []    # DynamicObject instances
    self.static_objects = []     # vapory object instances

  def add_object(self, obj):
    """Adds obj to the appropriate list, according to the type of the object.

    If obj is not an ExperimentObject or a vapory object, behavior is
    undefined.
    """
    if issubclass(type(obj), ExperimentObject):
      self.experiment_objects.append(obj)
    elif type(obj) == DynamicObject:
      self.dynamic_objects = []
    else:
      self.static_objects.append(obj)
  
  def set_camera(self, img_shape=None, camera_multiplier=None):
    """Sets the camera dimensions of the Experiment so that the output image has
    `img_shape`. If `img_shape` is not None, resets `self.img_shape`.
    """
    if img_shape != None:
      assert(len(img_shape) == 2)
      self.img_shape = tuple(img_shape)

    if camera_multiplier != None:
      assert(camera_multiplier > 0)
      self.camera_multiplier = camera_multiplier

    location = [0, 0, -self.img_shape[0]*self.camera_multiplier]
    look_at = [0,0,0]
    right = [self.img_shape[0] / self.img_shape[1], 0, 0]
    angle = 2*np.degrees(np.arctan(1 / (2*self.camera_multiplier)))
    
    # TODO: set camera projection matrix

    self.camera = vapory.Camera('location', location,
                                'look_at', look_at,
                                'right', right,
                                'angle', angle)

  def compute_mask(self):
    """Computes the label (annotation) for the scene, based on most recent vapory
    objects created.

    Returns the label.

    """
    mask = np.zeros(self.img_shape, dtype=np.uint8)
    
    return mask
    
  def compute_target(self, *args):
    """Calculate the target for this ExperimentObject. Called with the same args
    that The default behavior is to calculate no target, such as for experiments
    which only desire segmentation masks. Should be overwritten by subclasses.
    
    Target is always a vector. This can represent a classification or some
    numerical values (more likely).
    
    Returns a numpy array containing the targets for the most recent scene. If
    there are no targets, return None.
    
    """
    # TODO: necessary? Might also belong in ExperimentObject. Unclear.
    # TODO: calculate_target should be able to see all the args that were passed
    # into the vapory object, to determine any targets that it needs to, from
    # those properties.
    # TODO: associate target vector with each object in segmentation mask, how?
    
    return None
    
  def render_scene(self):
    """Renders a single scene, applying the various perturbations on each
    object/light source in the Experiment.

    Returns a dictionary: {"image" : img, "mask" : mask, "target" : target},
    where img, mask, and target are all tf.train.Features, ready to be written
    into a tfrecord.

    Calls the make_targets() function, implemented by subclasses, that uses
    the object locations, orientations, etc. set by render_scene, to calculate
    the targets.
    """

    dynamic_objects = [obj() for obj in self.dynamic_objects]
    experiment_objects = [obj() for obj in self.experiment_objects]
    all_objects = self.static_objects + dynamic_objects + experiment_objects
    scene = vapory.Scene(self.camera, all_objects, included=self.included)

    # img, mask ndarrays of np.uint8s.
    img = scene.render(height=self.img_shape[0], width=self.img_shape[1])
    mask = self.compute_mask()  # computes using most recently used args

    assert(img.ndim == 3 and mask.ndim == 3) # TODO: reshape img, mask if needed
    img_string = img.tostring()
    mask_string = lbl.tostring()
    image_shape = np.array(img.shape, dtype=np.int64)
    mask_shape = mask.shape(mask.shape, dtype=np.int64)

    return {"image" : _bytes_feature(img_string),
            "mask" : _bytes_feature(mask_string),
            "image_shape" : _int64_feature(image_shape),
            "mask_shape" : _int64_feature()}
    
  def run(self):
    """Generate the dataset as perscribed, storing it as self.fname. Should include
    the logic for each example, generating masks as well as target features.
    """
    writer = tf.python_io.TFRecordWriter(self.fname)
    
    for i in range(self.N):
      feature = self.render_scene()
      features = tf.train.Features(feature=feature)
      example = tf.train.Example(features=features)
      writer.write(example.SerializeToString())

    writer.close()
    
class BallExperiment(Experiment):
  """Generate an experiment with one or more balls.
  """
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def add_static_ball(self, center, radius, *args):
    """Adds a static ball to the scene.
    args:
      center: a list (or tuple) of length two or three, specifying the center of
        the ball. If len(center) == 2, ball is assumed to lie in image plane
        (z=0).
      radius: radius of the ball.

    TODO: add options to change color of ball.
    """
    center = list(center)
    if len(center) == 2:
      center.append(0)
    assert(len(center) == 3)
    ball = vapory.Sphere(center, radius, *args)
    self.add_object(ball)

#################### TESTS ####################
    
def test():
  color = lambda col: vapory.Texture(vapory.Pigment('color', col))
  
  exp = BallExperiment(img_shape=(512, 1024))
  exp.add_object(vapory.Background('White'))
  exp.add_object(vapory.LightSource([0, 500, -500], 'color', [1,1,1]))
  ball = ExperimentObject(vapory.Sphere, lambda : ((0,0), 50), color('Red'))
  exp.add_object(ball)
  feature = exp.render_scene()
  
  plt.imshow(feature['image'])
  plt.show()
    
def main():
  """For testing purposes"""
  test()
  

if __name__ == '__main__': main()
