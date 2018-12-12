"""experiments is a tool for creating large, labeled training sets for semantic
segmentation and/or object detection, with the ray-tracing tool POV-Ray.

Dependencies:
* numpy
* POV-Ray
* vapory

On masks and annotations:
A "mask" is a tuple of arrays, such as those returned by skimage.draw functions,
which index into the experiment's image space.
An "annotation" is an array with the same height and width as the experiment's image.

"""

import numpy as np
import vapory
import os
import matplotlib.pyplot as plt
from skimage import draw
from inspect import signature
import subprocess as sp

from artifice.utils import dataset, img


INFINITY = 10e9


def normalize(X):
  """Return normalized 1D vector X"""
  X = np.array(X)
  assert(len(X.shape) == 1)
  return X / np.linalg.norm(X)

def perpendicular(X):
  """Return a unit vector perpendicular to X in R^3."""
  X = np.array(X)
  assert(len(X.shape) == 1)
  return normalize(np.array([X[1] - X[2],
                             X[2] - X[0],
                             X[0] - X[1]]))

def quadratic_formula(a,b,c):
  """Return the two solutions according to the quadratic formula. If no
  real solutions exist, return None.
  """
  sqrt_term = b**2 - 4*a*c
  if sqrt_term < 0:
    return None

  return (-b + np.sqrt(sqrt_term)) / (2*a), (-b - np.sqrt(sqrt_term)) / (2*a)


class DynamicObject:
  """An ImageObject is a wrapper around a Vapory object. This may be used for any
  objects in a scene which change from image to image but are not being tracked.
  The vapory object itself is created on every __call__().

  args:
  * vapory_object: the vapory class this ExperimentObject represents.
  * object_args: either a tuple, containing all required arguments for creating
    vapory_object, or a function which creates this tuple (allowing for
    non-determinism), as well as any others that should change on each
    call. This function may optionally take an argument (such as a time step).
  * args: any additional args, passed on as vapory_object(*object_args + args)
  * kwargs: additions keyword arguments are passed onto the vapory object, the
    same on every call.

  self.args stores the most recent args as a list. This is instantiated as [],
  and is used by Experiment.compute_annotation() to generate the mask of the object.

  """
  def __init__(self, vapory_object, object_args, *args, **kwargs):
    self.vapory_object = vapory_object
    
    if callable(object_args):
      self.get_args = object_args
      self.do_time_step = len(signature(object_args).parameters) == 1
    elif type(object_args) == tuple:
      self.get_args = lambda : object_args
      self.do_time_step = False
    else:
      raise RuntimeError("`object_args` is not a tuple or function")

    self.other_args = list(args) + sum([list(t) for t in kwargs.items()], [])
    self.args = []

  def __call__(self, t=None):
    """Return an instance of the represented vapory object, ready to be inserted
    into a scene.

    TODO: allow arguments to be passed in, overriding get_args. Optional.
    """
    if self.do_time_step:
      assert(t is not None)
      self.args = list(self.get_args(t)) + self.other_args
    else:
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
    semantic_label: numerical class of the object, used for generating an
      annotation. default=1.
    args: any additional args, passed on as vapory_object(*object_args + args)
    kwargs: additions keyword arguments are passed onto the vapory object, the
      same on every call.

  """

  def __init__(self, vapory_object, object_args, *args, semantic_label=1, **kwargs):
    super().__init__(vapory_object, object_args, *args, **kwargs)
    assert(semantic_label > 0)
    self.semantic_label = int(semantic_label)

  def compute_mask(experiment):
    """Compute the mask of the ExperimentObject, given Experiment
    `experiment`.

    This should be overwritten by subclasses, for each type of vapory
    object. Each object returns the indices of the experiment scene that it
    contains (as in skimage.draw). It is up to the Experiment to decide, in the
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
    semantic_label: numerical class of the object, used for generating an
      annotation. default=1.
    args: any additional args, passed on as vapory_object(*object_args + args)
    kwargs: additions keyword arguments are passed onto the vapory object, the
      same on every call.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(vapory.Sphere, *args, **kwargs)
    self.center = self.radius = None

  def __call__(self, t=None):
    """Record the center and radius of the sphere."""
    vapory_object = super().__call__(t)
    self.center = np.array(self.args[0])
    self.radius = self.args[1]
    return vapory_object
  
  def distance_to_surface(self, Xi, experiment):
    """Given a point Xi = [x,y] in image-space, compute the distance from
    experiment.camera_location to the near-surface of the sphere.

    If Xi is not on the surface, return "infinity" (actually 1bil).
    """
    assert(len(self.args) != 0)
    const = experiment.camera_location - self.center
    v = experiment.unproject(Xi)

    a = np.linalg.norm(v)**2
    b = 2*np.dot(const, v)
    c = np.linalg.norm(const)**2 - self.radius**2

    ts = quadratic_formula(a,b,c)
    if ts == None:
      return INFINITY
    t1, t2 = ts

    # l = cam_loc + t*v, so d = distance to camera = l - cam_loc = t*v
    d1 = np.linalg.norm(t1*v)
    d2 = np.linalg.norm(t2*v)

    return min(d1, d2)

  def compute_mask(self, experiment):
    """Compute the mask for an ExperimentSphere, placed in experiment. Returns rr,
    cc, which are list of indices to access the image (as from skimage.draw),
    and dd: the distance from camera to object along each pixel in (rr,cc).

    """
    assert(len(self.args) != 0)
    center = experiment.project(self.center)
    center_to_edge = self.radius * perpendicular(
      experiment.camera_to(self.center))
    radius_vector = (experiment.project(self.center + center_to_edge)
                     - experiment.project(self.center))
    radius = np.linalg.norm(radius_vector)
    
    rr, cc = draw.circle(center[0], center[1], radius,
                         shape=experiment.image_shape[:2])
    
    dd = np.empty(rr.shape[0], dtype=np.float64)
    for i in range(dd.shape[0]):
      dd[i] = self.distance_to_surface([rr[i], cc[i]], experiment)
      
    return rr, cc, dd

  
class Experiment:
  """An Experiment contains information for generating a dataset, which is done
  using self.run(). It has variations that affect the output labels.
  
  args:
    image_shape: (rows, cols) shape of the output images, determines the aspect ratio
      of the camera, default=(512,512). Number of channels determined by `mode`
    mode: image mode to generate, default='L' (8-bit grayscale)
    num_classes: number of classes to be detected, INCLUDING the background class.
    N: number of images to generate, default=1000
    output_format: filetype to write, default='tfrecord'. Can be a list of
      filetypes, in which case the same data will be written to each.
    fname: name of output file, without extension. Ignored if included.
    camera_multiplier: controls how far out the camera is positioned, as a
      multiple of image_shape[1] (vertical pixels), default=4 (far away)

  Image `mode` is according to PIL.Image. Valid inputs are:
  * L (8-bit pixels, black and white)
  * RGB (3x8-bit pixels, true colour)
  Other modes to be supported later, including:

  The camera will be placed in each experiment such that the <x,y,0> plane is
  the image plane, with one unit of distance corresponding to ~1 pixel on that
  plane.

  self.objects is a list of ExperimentObjects that are subject to change,
  whereas self.static_objects is a list of vapory Objects ready to be inserted
  in the scene, as is.
  """

  supported_modes = {'L', 'RGB'}
  pix_fmts = {'L' : 'gray', 'RGB' : 'rgb8'}
  supported_formats = {'tfrecord', 'mp4'}
  included = ["colors.inc", "textures.inc"]

  def __init__(self, image_shape=(512,512), mode='L', num_classes=2, N=1000,
               output_format='tfrecord', fname="data", camera_multiplier=4,
               fps=1):
    self.N = int(N)

    assert(type(image_shape) == tuple and len(image_shape) == 2)
    self.image_shape = image_shape

    assert(mode in self.supported_modes)
    self.mode = mode

    # output formats
    if type(output_format) in [list, set]:
      self.output_formats = set(output_format)
    else:
      assert(type(output_format) == str)
      self.output_formats = {output_format}
    assert(all([f in self.supported_formats for f in self.output_formats]))

    # set fname, without extension
    assert(type(fname) == str)
    self.fname = fname

    assert(camera_multiplier > 0)
    self.camera_multiplier = camera_multiplier

    assert(num_classes > 0)
    self.num_classes = num_classes # TODO: unused
    self.fps = int(fps)
    
    self._set_camera()
    
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
  
  def _set_camera(self):
    """Sets the camera dimensions of the Experiment so that the output image has
    `image_shape`. Also sets the camera projection matrix. Should only be called
    by __init__().

    """

    camera_distance = self.image_shape[0]*self.camera_multiplier
    location = [0, 0, -camera_distance]
    direction = [0, 0, 1]       # POV-Ray direction vector
    aspect_ratio = self.image_shape[0] / self.image_shape[1] # aspect ratio
    right = [aspect_ratio, 0, 0]                             # POV-Ray vector
    half_angle_radians = np.arctan(1 / (2*self.camera_multiplier))
    
    # (Szeliski 53)
    focal_length = self.image_shape[1] / (2*np.tan(half_angle_radians))
    
    # Set the camera projection matrix.
    K = np.array(
      [[focal_length, 0, self.image_shape[0]/2],
       [0, aspect_ratio*focal_length, self.image_shape[1]/2],
       [0, 0, 1]])
    T = np.array(
      [[0],
       [0],
       [camera_distance]])
    R = np.array(
      [[0, -1, 0],
       [1, 0, 0],
       [0, 0, 1]])
    P = K @ np.concatenate((R, T), axis=1)
    self._camera_WtoI = np.concatenate((P, [[0, 0, 0, 1]]), axis=0)
    self._camera_ItoW = np.linalg.inv(self._camera_WtoI)

    self.camera_location = np.array(location)

    self.camera = vapory.Camera('location', location,
                                'direction', direction,
                                'right', right,
                                'angle', 2*np.degrees(half_angle_radians))

  def camera_to(self, X):
    """Get the world-space vector from the camera to X"""
    assert(len(X) == 3)
    return np.array(X) - self.camera_location
  
  def project(self, X):
    """Project the world-space POINT X = [x,y,z] to image-space.
    Return the [i,j] point in image-space (as a numpy array).
    """
    assert(len(X) == 3)
    Xi = self._camera_WtoI @ np.concatenate((np.array(X), [1]))
    return np.array([Xi[0]/Xi[2], Xi[1]/Xi[2]])

  def unproject_point(self, Xi, disparity=1):
    """From index space point Xi = [x,y], unproject back into world-space. Note
    that since an unambiguous 3D point cannot be recovered, this should be used
    only to recover a ray associated with a given pixel in image-space.

    The "disparity" argument controls this ambiguity. Different disparities will
    yield different points along the same ray.
    """
    assert(len(Xi) == 2)
    Xi = np.array(Xi)
    X = self._camera_ItoW @ np.array([Xi[0], Xi[1], 1, disparity])
    return (X / X[3])[:3]

  def unproject(self, Xi):
    """From index space point Xi = [x,y], unproject back into world-space. Due
    to 3D-2D ambiguity, an image-space corresponds to a ray in
    world-space. Returns a unit-vector along this ray. Together with camera
    location, this can recover any point along the ray.
    """
    Xi = np.array(Xi)
    a = self.unproject_point(Xi)
    b = self.unproject_point(Xi, disparity = 2)
    V = normalize(a - b)
    if V[2] == 0:
      return V
    else:
      return V * V[2] / abs(V[2]) # ensure V points toward +z

  def unproject_to_image_plane(self, Xi):
    """From index space point Xi = [x,y], unproject back to the world-space
    point which lies on the image plane.
    """
    Xi = np.array(Xi)
    u_hat = self.unproject(Xi)
    v = self.camera_location
    mag_v = np.linalg.norm(v)
    cos_th = np.dot(u,v) / (u_hat * mag_v)
    u = (mag_v / cos_th) * u_hat
    return v + u
  
  def compute_annotation(self):
    """Computes the annotation for the scene, based on most recent vapory
    objects created. Each annotation marks the class associated with the pixel.

    TODO: currently, only modifies masks due to occlusion by other objects in
    experiment_objects. This is usually sufficient, but in some cases, occlusion
    may occur from static or untracked objects.

    """
    annotation = np.zeros((self.image_shape[0], self.image_shape[1], 1),
                          dtype=np.uint8)
    object_distance = INFINITY * np.ones(annotation.shape[:2], dtype=np.float64)
    
    for obj in self.experiment_objects:
      rr, cc, dd = obj.compute_mask(self)
      for i in range(rr.shape[0]):
        if dd[i] < object_distance[rr[i], cc[i]]:
          object_distance[rr[i], cc[i]] = dd[i]
          annotation[rr[i], cc[i], 0] = obj.semantic_label

    return annotation
    
  def compute_target(self, *args):
    """Calculate the target for this ExperimentObject. Called with the same args
    that The default behavior is to calculate no target, such as for experiments
    which only desire segmentation annotations. Should be overwritten by subclasses.
    
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
    
  def render_scene(self, t=None):
    """Renders a single scene, applying the various perturbations on each
    object/light source in the Experiment.

    Returns a "scene", tuple of `image` and `annotation`.

    TODO:
    Call the make_targets() function, implemented by subclasses, that uses
    the object locations, orientations, etc. set by render_scene, to calculate
    the targets.
    """

    dynamic_objects = [obj(t) for obj in self.dynamic_objects]
    experiment_objects = [obj(t) for obj in self.experiment_objects]
    all_objects = self.static_objects + dynamic_objects + experiment_objects
    scene = vapory.Scene(self.camera, all_objects, included=self.included)

    # image, annotation ndarrays of np.uint8s.
    image = scene.render(height=self.image_shape[0], width=self.image_shape[1])
    if self.mode == 'L':
      image = img.grayscale(image)
    annotation = self.compute_annotation()  # computes using most recently used args

    return image, annotation
    
  def run(self, verbose=False):
    """Generate the dataset in each format.
    """

    if len(self.output_formats) == 0:
      # TODO: raise error?
      return
    
    # Instantiate writers and fnames for each format
    if 'tfrecord' in self.output_formats:
      tfrecord_fname = self.fname + '.tfrecord'
      tfrecord_writer = tf.python_io.TFRecordWriter(fname)
      if verbose:
        print("Writing tfrecord to {}...".format(mp4_image_fname))

    if 'mp4' in self.output_formats:
      mp4_image_fname = self.fname + '.mp4'
      mp4_annotation_fname = self.fname + '_annotation.mp4'

      if verbose:
        print("Writing video to {}...".format(mp4_image_fname))
      image_cmd = [
        'ffmpeg',
        '-y',              # overwrite existing files
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', '{}x{}'.format(*self.image_shape), # frame size
        '-pix_fmt', self.pix_fmts[self.mode],
        '-r', str(self.fps), # frames per second
        '-i', '-',           # input comes from a pipe
        '-an',               # no audio
        '-vcodec', 'mpeg4',
        mp4_image_fname]
      annotation_cmd = [
        'ffmpeg',
        '-y',              # overwrite existing files
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', '{}x{}'.format(*self.image_shape), # frame size
        '-pix_fmt', 'rgba',
        '-r', str(self.fps), # frames per second
        '-i', '-',           # input comes from a pipe
        '-an',               # no audio
        '-vcodec', 'mpeg4',
        mp4_annotation_fname]      
      mp4_annotation_cmap = plt.get_cmap('tab20c', lut=self.num_classes)

      mp4_image_log = open(self.fname + '_log.txt', 'w+')
      mp4_annotation_log = open(self.fname + '_annotation_log.txt', 'w+')
      mp4_image_proc = sp.Popen(image_cmd, stdin=sp.PIPE, stderr=mp4_image_log)
      mp4_annotation_proc = sp.Popen(annotation_cmd, stdin=sp.PIPE,
                                     stderr=mp4_annotation_log)

    # step through all the frames, rendering each scene with time-dependence if
    # necessary.
    for t in range(self.N):
      if verbose:
        print("Rendering scene {} of {}...".format(t, self.N))
      scene = self.render_scene(t)
      image, annotation = scene
      
      if 'tfrecord' in self.output_formats:
        e = dataset.example_string_from_scene(*scene)
        tfrecord_writer.write(e)
        
      if 'mp4' in self.output_formats:
        mp4_image_proc.stdin.write(image.tostring())
        annotation_map = mp4_annotation_cmap(annotation[:,:,0]) * 255
        annotation_map = annotation_map.astype(np.uint8)
        mp4_annotation_proc.stdin.write(annotation_map.tostring())

    if 'tfrecord' in self.output_formats:
      tfrecord_writer.close()
    if 'mp4' in self.output_formats:
      mp4_image_proc.stdin.close()
      mp4_image_proc.wait()
      mp4_image_log.close()
      
      mp4_annotation_proc.stdin.close()
      mp4_annotation_proc.wait()
      mp4_annotation_log.close()

      
class BallExperiment(Experiment):
  """Generate an experiment with one or more balls. Mostly for test cases.

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

def annotation_diff(image, annotation):
  """Assuming annotation has a black background, return a numpy array of binary
  values that shows which pixels are off."""
  bin_image = np.not_equal(image, 0).any(axis=2)
  bin_annotation = np.not_equal(annotation,0).reshape(annotation.shape[:2])
  return np.not_equal(bin_image, bin_annotation)

def test():
  color = lambda col: vapory.Texture(vapory.Pigment('color', col))
  width = 500
  
  exp = BallExperiment(image_shape=(width, width))
  exp.add_object(vapory.Background('Black'))
  exp.add_object(vapory.LightSource([0, 5*width, -5*width], 'color', [1,1,1]))
  # TODO: oboe with radius?
  min_radius = 20
  max_radius = 100
  argsf = lambda : (
    [np.random.randint(max_radius/2,width/2 - max_radius/2),
     np.random.randint(max_radius/2,width/2 - max_radius/2),
     np.random.randint(-width, width)
    ],
    np.random.randint(min_radius, max_radius))
  
  red_ball = ExperimentSphere(argsf, color('Red'))
  blue_ball = ExperimentSphere(argsf, color('Blue'), semantic_label=2)
  exp.add_object(red_ball)
  exp.add_object(blue_ball)
  
  image, annotation = exp.render_scene()

  # world_point = red_ball.center
  # image_point = exp.project(*world_point)
  
  plt.imshow(image)
  # plt.plot(image_point[0], image_point[1], 'b.')
  plt.savefig('image.png')

  plt.imshow(annotation.reshape(annotation.shape[:2]))
  plt.savefig('annotation.png')

  plt.imshow(annotation_diff(image, annotation))
  plt.savefig('annotation_diff.png')
    
def main():
  """For testing purposes"""
  test()
  

if __name__ == '__main__': main()
