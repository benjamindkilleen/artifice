"""experiments is a tool for creating large, labeled training sets for semantic
segmentation and/or object detection, with the ray-tracing tool POV-Ray.

Dependencies:
* numpy
* POV-Ray
* vapory
* tensorflow
"""

import numpy as np
import vapory
import os
import tensorflow as tf

class ExperimentObject:
  """An ExperimentObject represents the objects which may appear in an
  Experiment. The vapory object itself is created on __call__().

  This class is intended to fully encompass all needed vapory
  objects. Subclassing is desirable when many of the same object are present in
  a scene, each with their own location/orientation attributes.
  """

  def __init__(self, vapory_object, object_args, **kwargs):
    """
    args:
      vapory_object: the vapory class this ExperimentObject represents.
      object_args: either a tuple, containing all required arguments for creating
        vapory_object, or a function which creates this tuple (allowing for
        non-determinism), as well as any others that should change on each call.
      kwargs: additions keyword arguments are passed onto the vapory object, the
        same on every call.
    """

    self.vapory_object = vapory_object
    
    if callable(object_args):
      self.get_args = object_args
    elif type(object_args) == tuple:
      self.get_args = lambda : object_args
    else:
      raise RuntimeError("`object_args` is not a tuple or function")

    self.other_args = sum([list(t) for t in kwargs.items()], [])

  def __call__(self):
    """Return an instance of the represented vapory object, ready to be inserted
    into a scene.
    """
    args = self.get_args()
    return self.vapory_object(*args + self.other_args)
    
class Experiment:
  """An Experiment contains information for generating a dataset, which is done
  using self.run(). It has variations that affect the output labels.

  By default, an Experiment will write 1000 examples to a tfrecord file,
  storing images as 8-bit grayscale.

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

  self.included is a list of POV-Ray files to include.

  Brainstorm: Each Experiment will have a list of Vapory objects in each scene,
  which includes both "objects" and light sources. Each of these needs to
  somehow determine where it can appear, as a function of which image it is in
  the dataset (to allow for temporal dependent datasets). Some objects may
  appear always, but others may only appear sometimes (background clutter,
  occlusion, etc.).

  """

  supported_modes = {'L', 'RGB'}
  supported_formats = {'tfrecord'}

  def __init__(self, img_shape=(512,512), mode='L', N=1000, output_format='tfrecord',
               fname="out"):
    """
    args:
      img_shape: (rows, cols) shape of the output images, determines the aspect ratio
        of the camera, default=(512,512). Number of channels determined by `mode`
      mode: image mode to generate, default='L' (8-bit grayscale)
      N: number of images to generate, default=1000
      output_format: filetype to write, default='tfrecord'
      fname: name of output file, without extension. Ignored if included.
    """
    assert(type(img_shape) == tuple and len(img_shape) == 2)
    assert(mode in self.supported_modes)
    assert(output_format in self.supported_formats)
    assert(fname == str)
    fname = '.'.join(fname.split('.')[:-1])
    
    self.img_shape = img_shape
    self.N = int(N)
    self.mode = mode
    self.output_format=output_format
    self.fname = fname

    self.set_camera()

    # The objects in the scene should be added to by the subclass.
    self.experiment_objects = [] 
    self.static_objects = []
    self.included = ["colors.inc", "textures.inc"]
    # TODO: make methods to add or delete objects

  def add_object(obj):
    """Adds obj to the appropriate list, according to the type of the object.

    If obj is not an ExperimentObject or a vapory object, behavior is
    undefined.
    """
    if type(obj) == ExperimentObject:
      self.experiment_objects.append(obj)
    else:
      self.static_objects.append(obj)
  
  def set_camera(img_shape=None):
    """Sets the camera dimensions of the Experiment so that the output image has
    `img_shape`. If `img_shape` is not None, resets `self.img_shape`.
    """
    if img_shape != None:
      assert(type(img_shape) == tuple and len(img_shape) == 2)
      self.img_shape = img_shape

    location = [0, 0, -self.img_shape[1]]
    look_at = [0,0,0]
    right = [self.img_shape[0] / self.img_shape[1], 0, 0]

    self.camera = vapory.Camera('location', location,
                                'look_at', look_at,
                                'right', right)

  def calculate_targets(self):
    """Calculate the targets for this Experiment. The default behavior is to
    calculate no targets, such as for experiments which only desire segmentation
    masks.

    Targets is always a vector. This can represent a classification or some
    numerical values (more likely).

    Returns a numpy array containing the targets for the most recent scene. If
    there are no targets, return None.
    """

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

    mask, targets = None # TODO: placeholder

    scene_objects = [obj() for obj in self.experiment_objects] + self.static_objects
    scene = vapory.Scene(self.camera, scene_objects, included=self.included)
    img = scene.render(height=self.size[0], width=self.size[1])

    print(type(img))
    exit()
    
    return {"image" : img, "mask" : mask, "target" : target}
    
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
  
  def __init__(self, num_balls=1, **kwargs):
    super().__init__(**kwargs)

    assert(num_balls > 0)
    self.num_balls = int(num_balls)

  def add_static_ball(self, center, radius):
    """Adds a static red ball to the scene.
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
    ball = vapory.Sphere(center, radius, Pigment('Red'))
    self.add_object(ball)

#################### TESTS ####################
    
def center_test():
  exp = BallExperiment()
  exp.add_static_ball((0,0), 10)
  exp.render_scene()
    
def main():
  """For testing purposes"""
  center_test()
  

if __name__ == '__main__': main()
