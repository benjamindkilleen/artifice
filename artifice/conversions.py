"""Module for converting data to standard form expected by Artifice.

To create custom conversions, create a new function in the style of
`png_dir_and_npy_file`, the name of which specifies that the iamges have a
'.png' extension and the labels be in a single 'labels.npy' file in the
data_root. A conversion function should return nothing, merely saving the
resulting dataset to a location expected by Artifice, usually
data_root/labeled_set.tfrecord or data_root/unlabeled_set.tfrecord.

"""

from os.path import join, splitext
from glob import glob
import logging
import numpy as np
import tensorflow as tf
from artifice import img, dat

logger = logging.getLogger('artifice')

def _get_paths(dirpath, ext):
  paths = sorted(glob(join(dirpath, f'*.{ext}')))
  if not paths:
    raise FileNotFoundError("no '.{ext}' files in {dirpath}")
  return paths

def _load_single_labels(labels_path):
  """Load labels from a single file."""
  ext = splitext(labels_path)[1]
  raise NotImplementedError

def _image_dir_and_label_file(data_root, record_name='labeled_set.tfrecord',
                              image_dirname='images', image_ext='png',
                              labels_filename='labels.npy'):
  """Helper function to performs the conversion when labels are in one file.

  Assumes fully labeled data.

  :param data_root: root directory
  :param image_dirname: name of directory containing images by path
  :param image_ext: extension of images, e.g. 'png', 'jpeg'
  :param labels_ext: extension of data_root/labels.{ext} file.

  """
  image_paths = _get_paths(join(data_root, image_dirname), image_ext)
  labels = _load_single_labels(labels_path)
  raise NotImplementedError

_label_loaders = {'npy' : np.load,
                  'txt' : np.loadtxt}

def _image_dir_and_label_dir(data_root, record_name='labeled_set.tfrecord',
                             image_dirname='images', image_ext='png',
                             label_dirname='labels', label_ext='npy',
                             num_parallel_calls=None):
  """Performs the conversion when labels in corresponding files.

  :param data_root:
  :param image_dirname:
  :param image_ext:
  :param label_dirname:
  :param label_ext:
  :returns:
  :rtype:

  """
  image_paths = _get_paths(join(data_root, image_dirname), image_ext)
  label_paths = _get_paths(join(data_root, label_dirname), label_ext)
  if len(image_paths) != len(label_paths):
    raise RuntimeError(f"number of images ({len(image_paths)}) != "
                       f"number of labels ({len(label_paths)})")
  logger.info(f"writing {len(image_paths)} examples to "
              f"{join(data_root, record_name)}...")
  dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
  def _read_and_serialize(image_path, label_path):
    image = img.open_as_float(image_path)
    label = _label_loaders[label_ext](label_path)
    return dat._proto_from_example((image, label))
  dataset = dataset.map(
    lambda image_path, label_path : tuple(tf.py_function(
      _read_and_serialize, [image_path, label_path], [tf.string])),
    num_parallel_calls=num_parallel_calls)
  dat.save_dataset(join(data_root, record_name), dataset)

def png_dir_and_txt_dir(data_root, num_parallel_calls=None):
  """Convert from a directory of image files and dir of label files.

  Expect DATA_ROOT/images/ containing loadable images all of the same
  form and DATA_ROOT/labels/ with corresponding labels in text files, which can
  be loaded with np.loadtxt.

  """
  _image_dir_and_label_dir(data_root, image_ext='png', label_ext='txt',
                           num_parallel_calls=num_parallel_calls)

def png_dir_and_txt_file(data_root, num_parallel_calls=None):
  """Convert from a directory of image files, along with a `labels.txt` file.

  Expect DATA_ROOT/images/ directory containing loadable images all of the same
  form and DATA_ROOT/labels.txt with corresponding labels as can be loaded by
  np.loadtxt().

  """
  raise NotImplementedError

def png_dir_and_npy_dir(data_root, num_parallel_calls=None):
  """Convert from a directory of image files, along with a `labels.npy` file.

  Expect DATA_ROOT/images/ directory containing loadable images all of the same
  form and DATA_ROOT/labels/ with corresponding labels .npy files.

  """
  _image_dir_and_label_dir(data_root, image_ext='png', label_ext='npy',
                           num_parallel_calls=num_parallel_calls)

def png_dir_and_npy_file(data_root, num_parallel_calls=None):
  """Convert from a directory of image files, along with a `labels.npy` file.

  Expect DATA_ROOT/images/ directory containing loadable images all of the same
  form and DATA_ROOT/labels.npy with corresponding labels in one numpy array.

  """
  raise NotImplementedError

# list of all the conversion functions here.
conversions = [png_dir_and_txt_dir,
               png_dir_and_txt_file,
               png_dir_and_npy_dir,
               png_dir_and_npy_file]
