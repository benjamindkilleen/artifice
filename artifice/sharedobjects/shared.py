from os.path import exists
import pickle
from filelock import FileLock


class SharedDict(dict):
  """Maintain a dict. Doesn't allow reads or writes unless the dict is
  acquired. If path already exists, does not clear it unless clear() method is
  called.

  """

  def __init__(self, path):
    self.path = path
    if not exists(path):
      self._save()
    self.lock = FileLock(self.path + ".lockfile")

  def _check_acquired(self):
    if not self.lock.is_locked:
      raise RuntimeError("SharedDict: call acquire() before accessing dict.")

  def __getitem__(self, key):
    self._check_acquired()
    return super().__getitem__(key)

  def __setitem__(self, key, val):
    self._check_acquired()
    return super().__setitem__(key, val)

  def _save(self):
    with open(self.path, 'wb') as f:
      pickle.dump(super().copy(), f)

  def _load(self):
    self.clear()
    with open(self.path, 'rb') as f:
      self.update(pickle.load(f))

  def acquire(self, *args, **kwargs):
    """Acquire this dictionary in order to modify it.

    """
    self.lock.acquire(*args, **kwargs)
    self._load()

  def release(self, *args, **kwargs):
    """Release this dictionary, allowing other processes to acquire it.

    """
    self._save()
    self.lock.release(*args, **kwargs)
