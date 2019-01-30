"""Transformation utils, used to build up augmentations."""

from artifice.utils import inpaint, tfimg
import tensorflow as tf
import logging

logger = logging.getLogger('artifice')

class Transformation():
  """A transformation is a callable that takes tensors representing an example
  (usually a scene) and returns a new pair. It should be mappable over a
  tf.data.Dataset.

  Transformations are meant to be aggressively subclassed. This allows families
  of transformations which, although not totally identitcal, to belong
  together. These can define the transform as a method rather than passing it in
  on initialization.

  Transformations can be composed (or "added together") using the "+"
  operator.

  Rather than pass in a function, subclasses may optionally define a "transform"
  method, which is taken at instantiation. This is supported but not preferred.

  """

  def __init__(self, transform=lambda scene : scene, **kwargs):
    """
    :param transforms: a transformation function or an iterable of them. Ignored if
      object has a "transform" method.
    :param which_examples: indices of examples in the dataset on which to apply
      the transformation. If None, maps over the whole dataset. 
    
    """

    if hasattr(self, 'transform'):
      assert callable(self.transform)
      self._transforms = [lambda scene : self.transform(scene)]
    elif callable(transform):
      self._transforms = [transform]
    elif hasattr(transform, '__iter__'):
      self._transforms = list(transform)
    else:
      raise ValueError()

    self.which_examples = kwargs.get('which_examples')
    if (self.which_examples is not None 
        and type(self.which_examples) != tf.Tensor):
      self.which_examples = tf.constant(self.which_examples, tf.int64)

    self._kwargs = kwargs

  def __call__(self, scene):
    for transform in self._transforms:
      scene = transform(scene)
    return scene

  def apply(self, dataset, num_parallel_calls=None):
    if self.which_examples is None:
      return dataset.map(self, num_parallel_calls=num_parallel_calls)
    
    enumerated = dataset.apply(tf.contrib.data.enumerate_dataset())
    predicate = lambda idx, _ : tf.reduce_any(tf.equal(idx, self.which_examples))
    filtered = enumerated.filter(predicate)
    return filtered.map(
      lambda _, scene : self.__call__(scene), num_parallel_calls=num_parallel_calls)

  def __add__(self, other):
    return Transformation(self._transforms + other._transforms)

  def __radd__(self, other):
    if other == 0:
      return self
    else:
      return self.__add__(other)


"""For many transformations, Simple- and ImageTransformations should be
sufficient, and they may instantiated with transform functions on their own,
depending on whether that transform should applied to both image and annotation
(SimpleTransformation) or to the image alone (ImageTransformation)

Transformations that treat image and annotation separately should inherit from
Transformation directly.
"""
class SimpleTransformation(Transformation):
  """Applies the same tensor function to both image and annotation, clipping image
  values. Applies a different transform function to the labels

  """
  def __init__(self, image_transform, label_transform):
    def transform(scene):
      image, (annotation, label)
      image = image_transform(image)
      image = tf.clip_by_value(image,
                               tf.constant(0, dtype=image.dtype),
                               tf.constant(1, dtype=image.dtype))
      return image, (image_transform(annotation), label_transform(label))
    super().__init__(transform)

class ImageTransformation(Transformation):
  """Applies a tensor function to the image (and clips values), leaving annotation
  and label.

  """
  def __init__(self, function):
    def transform(scene):
      image, (annotation, label) = scene
      image = function(image)
      image = tf.clip_by_value(image,
                               tf.constant(0, dtype=image.dtype),
                               tf.constant(1, dtype=image.dtype))
      return image, (annotation, label)
    super().__init__(transform)


class FlipLeftRight(Transformation):
  def __init__(self):
    def transform(scene):
      image, (annotation, label) = scene
      image = tf.flip_left_right(image)
      annotation = tf.flip_left_right(annotation)
      label[:,2] = image.shape[1] - label[:,2]
      return image, (annotation, label)
    super().__init__(transform)
      

class FlipUpDown(Transformation):
  def __init__(self):
    def transform(scene):
      image, (annotation, label) = scene
      image = tf.flip_up_down(image)
      annotation = tf.flip_up_down(annotation)
      label[:,1] = image.shape[0] - label[:,1]
      return image, (annotation, label)
    super().__init__(transform)


"""The following are transformations that must be instantiated with a
parameter separate from the arguments given to the transformation function."""
class AdjustBrightness(ImageTransformation):
  """Adjust the brightness of the image by delta."""
  def __init__(self, delta):
    def transform(image):
      return tf.image.adjust_brightness(image, delta)
    super().__init__(transform)


class AdjustMeanBrightness(ImageTransformation):
  """Adjust the mean brightness of grayscale images to mean_brightness. Afterward,
  clip values as appropriate. Thus the final mean brightness might not be the
  value passed in. Keep this in mind. 

  """
  def __init__(self, new_mean): 
    def transform(image):
      mean = tf.reduce_mean(image)
      delta = tf.constant(new_mean, dtype=mean.dtype) - mean
      return tf.image.adjust_brightness(image, delta)
    super().__init__(transform)


"""The following are transformations that introduce truly novel examples by
extracting and then re-inserting examples."""
class ObjectTransformation(Transformation):
  def __init__(self, new_label, **kwargs):
    self.new_label = new_label
    self.name = kwargs.get('name', self.__class__.__name__)
    self.inpainter = kwargs.get('inpainter', inpaint.background)
    self.background_image = kwargs.get('background_image')
    self.object_order = kwargs.get('object_order', range(new_label.shape[0]))
    self.num_classes = kwargs.get('num_classes', 2)
    self.num_objects = kwargs.get('num_objects', 10)
    super().__init__(**kwargs)

  def transform(self, scene):
    """Transforms `scene` to match `self.new_label`.

    :param scene: (image, (annotation, label)) tensor tuple
    :returns: transformed scene
    :rtype: tuple of tensors

    """
    image, (annotation, label) = scene
    new_image = tf.Variable(tf.identity(image), validate_shape=False)
    new_annotation = tf.Variable(tf.identity(annotation), validate_shape=False)
    
    components = tfimg.connected_components(annotation, num_classes=self.num_classes)
    component_ids = tf.constant(
      False, tf.bool, [self.num_classes, self.num_objects])
    
    for i in self.object_order:
      indices = tfimg.connected_component_indices(
        annotation, label[i,0], label[i, 1:3],
        num_classes=self.num_classes,
        components=components,
        component_ids=None)     # TODO: keep track of component_ids
      if indices is None:
        continue;

      new_indices, image_values, annotation_values = self.transform_image(
        image, annotation, label[i], self.new_label[i], num_classes=self.num_classes)

      # TODO: use self.inpainter
      new_image = inpaint.background(new_image, new_indices,
                                     background_image=self.background_image)
      new_image = tf.scatter_nd_update(new_image, new_indices, 
                                       image_values, name=self.name)
      
      new_annotation = inpaint.annotation(new_annotation, indices)
      new_annotation = tf.scatter_nd_update(new_annotation, new_indices, 
                                            annotation_values, 
                                            name='annotation_' + self.name)
    
    return new_image, (new_annotation, self.new_label)


  @staticmethod
  def transform_image(image, annotation, obj_label, new_obj_label,
                      num_classes=2):
    """Get the new indices and values for transforming between obj_label and
    new_obj_label.

    1. Translate object to image/annotation center.
    2. Apply a bilinear rotation to image. Apply nearest-neighbor rotation to
       annotation.
    3. Apply bilinear resizing to image. Apply NN resize to annotation.
    4. Translate back.
    5. Get components, indices, etc, as above.

    :param image: 
    :param annotation: 
    :param obj_label: 
    :param new_obj_label: 
    :returns: (indices, values) for the transformed object, into the original image.
    :rtype: tuple of tensors

    """
    centering = -obj_label[1:3]   # Get translation to center.
    centered_image = tf.contrib.image.translate(
      image, centering, 'BILINEAR')
    centered_annotation = tf.contrib.image.translate(
      annotation, centering, 'NEAREST')

    logger.debug(f"new_obj_label: {new_obj_label}")
    logger.debug(f"obj_label: {obj_label}")
    rotation = new_obj_label[3] - obj_label[3]
    rotated_image = tf.contrib.image.rotate(
      centered_image, rotation, 'BILINEAR')
    rotated_annotation = tf.contrib.image.rotate(
      centered_annotation, rotation, 'NEAREST')

    logger.debug(f"image: {image}")
    size = tf.to_int32(new_obj_label[4:6] / obj_label[4:6] *
                       tf.to_float(tf.shape(image)[:2])) 
    resized_image = tf.image.resize_images(
      rotated_image, size, tf.image.ResizeMethod.BILINEAR)
    resized_annotation = tf.image.resize_images(
      rotated_annotation, size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    location = new_obj_label[1:3]
    translated_image = tf.contrib.image.translate(
      resized_image, location, 'BILINEAR')
    translated_annotation = tf.contrib.image.translate(
      resized_annotation, location, 'NEAREST')

    new_indices = tfimg.connected_component_indices(
      annotation, obj_label[0], location,
      num_classes=num_classes)
    new_indices = tfimg.get_inside(new_indices, image)
    image_values = tf.gather_nd(image, new_indices)
    annotation_values = ObjectTransformation.get_annotation_values( 
      new_indices, new_obj_label)
  
    return new_indices, image_values, annotation_values

  @staticmethod
  def get_annotation_values(new_indices, new_obj_label):
    """Get the values for the new annotation.

    :param new_indices: indices of the new object
    :param new_obj_label: label for the new object
    :returns: new annotation_values
    :rtype: 

    """
    
    differences = (tf.to_float(new_indices) -
                   tf.reshape(new_obj_label[1:3], [1,2]))
    distances = tf.norm(tf.to_float(differences), axis=1, keepdims=True)
    semantic_labels = tf.to_float(tf.fill(tf.shape(distances), new_obj_label[0]))
    return tf.concat((semantic_labels, distances), axis=1)


# TODO: ObjectTranslation, ObjectRotation, and ObjectScaling subclassed from
# ObjectTransformation. The point is mostly moot, since ObjectTransformation
# generalizes over all of them, but some granularity might be useful in the
# future.
class ObjectTranslation(ObjectTransformation):
  def __init__(self):
    raise NotImplementedError

class ObjectRotation(ObjectTransformation):
  def __init__(self):
    raise NotImplementedError

class ObjectScaling(ObjectTransformation):
  def __init__(self):
    raise NotImplementedError


# Transformation instances.
identity = Transformation()
flip_left_right = FlipLeftRight()
flip_up_down = FlipUpDown()
invert_brightness = ImageTransformation(lambda image : 1 - image)


