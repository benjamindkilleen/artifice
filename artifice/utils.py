"""Generic utils."""

import os
import json
import shutil
import logging

logger = logging.getLogger('artifice')

def listwrap(val):
  """Wrap `val` as a list.

  :param val: iterable or constant
  :returns: `list(val)` if `val` is iterable, else [val]
  :rtype: 

  """
  if isinstance(val, list):
    return val
  return [val]


def listify(val, length):
  """Ensure `val` is a list of size `length`.

  :param val: iterable or constant
  :param length: integer length
  :returns: listified `val`.
  :raises: RuntimeError if 1 < len(val) != length

  """
  if not isinstance(val, str) and hasattr(val, '__iter__'):
    val = list(val)
    if len(val) == 1:
      return val * length
    if len(val) != length:
      raise RuntimeError("mismatched length")
    return val
  return [val] * length

def jsonable(hist):
  """Make a history dictionary json-serializable.

  :param hist: dictionary of lists of float-like numbers.

  """
  out = {}
  for k,v in hist.items():
    out[k] = list(map(float, v))
  return out

def json_save(fname, obj):
  """Opens json object stored at fname. Errors if file doesn't exist."""
  with open(fname, 'w') as f:
    f.write(json.dumps(obj))

def json_load(fname):
  with open(fname, 'r') as f:
    obj = json.loads(f.read())
  return obj

def atleast_4d(image):
  """Expand a numpy array (typically an image) to 4d.

  Inserts batch dim, then channel dim.

  :param image: 
  :returns: 
  :rtype: 

  """
  if image.ndim >= 4:
    return image
  if image.ndim == 3:
    return image[np.newaxis, :, :, :]
  if image.ndim == 2:
    return image[np.newaxis, :, :, np.newaxis]
  if image.ndim == 1:
    return image[np.newaxis, :, np.newaxis, np.newaxis]

def rm(path):
  if not os.path.exists(path):
    return
  if os.path.isfile(path):
    os.remove(path)
  elif os.path.isdir(path):
    shutil.rmtree(path)
  else:
    raise RuntimeError(f"bad path: {path}")
  logger.info(f"removed {path}.")
