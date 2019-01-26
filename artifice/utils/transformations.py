"""Transformation utils, used to build up augmentations."""

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

  def __init__(self, transform=lambda scene : scene):
    """
    :transforms: a transformation function or an iterable of them. Ignored if
      object has a "transform" method.
    """
    if hasattr(self, 'transform'):
      assert callable(self.transform)
      self._transforms = [lambda *scene : self.transform(*scene)]
    elif callable(transform):
      self._transforms = [transform]
    elif hasattr(transform, '__iter__'):
      self._transforms = list(transform)
    else:
      raise ValueError()

  def __call__(self, scene):
    for transform in self._transforms:
      scene = transform(scene)
    return scene

  def apply(self, dataset, num_parallel_calls=None):
    return dataset.map(self, num_parallel_calls=num_parallel_calls)

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
  """ObjectTransformations are meant to be applied in a single pass-through of the
  data and then saved. They are intended to be targeted at specific examples,
  although this is not required, rather than to every example in the dataset.

  """
  def __init__(self, new_label, **kwargs):
    self.name = kwargs.get('name', self.__class__.__name__)
    self.inpainter = kwargs.get('inpainter', inpaint.background)
    self.object_order = kwargs.get('object_order', range(label.shape[0]))
    self.num_classes = kwargs.get('num_classes', 2)
    super().__init__()

  def transform(self, scene):
    image, (annotation, label) = scene
    new_image = tf.identity(image)
    new_annotation = tf.identity(annotation)
    
    components = tfimg.connected_components(annotation, num_classes=self.num_classes)
    component_ids = [set() for _ in range(self.num_classes)]
    
    for i in self.object_order:
      semantic_class = tf.cast(label[i,0], tf.int64)
      x = tf.cast(label[i,1], tf.int64)
      y = tf.cast(label[i,2], tf.int64)
      component_id = components[x, y, semantic_class]
      if component_id in components[semantic_class]:
        continue;             # component already encountered
      else:
        component_ids[semantic_class].add(component_id)

      indices = tf.where(tf.equal(components[:,:,semantic_class], component_id))
      new_indices, new_values = self.transform_indices(
        image, indices, label, new_label)

      new_image = self.inpainter(new_image, **kwargs)
      new_image = scatter_update(new_image, new_indices, 
                                 image_values, name=self.name)
      
      annotation_values = tf.cast(semantic_class, annotation.dtype)
      new_annotation = inpaint.annotation(new_annotation, indices)
      new_annotation = scatter_update(new_annotation, new_indices, 
                                      annotation_values, 
                                      name='annotation_' + self.name)
      
    return new_image, (new_annoation, new_label)


  @staticmethod
  def transform_indices(image, indices, label, new_label):
    # 1. Translate indices to center. This is easy, introduces no new points.
    # 2. Apply a rotation to each index to bring it to the new theta. The
    #    indices at this point.
    # 3. Apply a scaling to these indices.
    # 4. So now I have a list of index-space points as well as the values
    #    associated with them, which I have to sample, presumably with
    #    continuous convolution. That seems annoying. Is there a better way to
    #    do this, maybe with built-in tf stuff? Do some research.
    # 5. Translate to new location.
    # Other idea: just abandon tensorflow for this part of the augmentation, no
    # real reason to do this with tensorflow, except maybe that it's faster. If
    # there's any built-in stuff, then that's a different story, but that might
    # not be the case, and skimage definitely has some good stuff for this,
    # without my having to go through a bunch of headaches.
    new_indices = tf.cast(indices, tf.float64)
    values = tf.gather_nd(image, indices)
    return indices, values
    
    
  @staticmethod
  def translate_indices(indices, label, new_label):
    # This is where, with the potential for abstraction, the actual object
    # transformations are applied, to the indices of the connected component.
    # TODO: in progress
    location = label[i, 1:3]
    new_location = new_label[i, 1:3]
    indices_update = tf.cast(
      new_location - location, tf.int64).reshape([1,1,2])
    new_indices = tfimg.get_inside(indices + indices_update, image)
    

class Translation(ObjectTransformation):
  """Translate objects in the image so that their new locations match the
  locations in new_label. This involves a new translation for each object, in
  the order of the new labeling. Inpaints with the given inpainter.

  :new_label: required
  :inpainter: background to inpaint with
  :object_order: iterable permutation of indices into new_label, determining
    order of drawn translated object, back-to-front. 
    Default: same order as labeling.
  :num_classes:
  :name: name of the operation, default: translation

  Additional keyword args passed to inpainter as appropriate.

  """

    super().__init__(transform)



# Transformation instances.
identity_transformation = Transformation()
flip_left_right_transformation = FlipLeftRight()
flip_up_down_transformation = FlipUpDown()
invert_brightness_transformation = ImageTransformation(lambda image : 1 - image)


