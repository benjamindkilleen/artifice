"""Generic utils."""

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

