"""Make a nice little video writer."""

import numpy as np
import subprocess as sp
import logging

logger = logging.getLogger('artifice')


class MP4Writer:
  def __init__(self, fname, shape, fps=30):
    """Opens a process to start writing frames.

    :param fname: file to write the mp4 to
    :param shape: shape of the video. For rgb, include number of channels.
    :param fps: frames per second. Default is 30

    """
    self.fname = fname
    self.shape = tuple(shape)
    self.fps = fps
    if len(self.shape) == 2:
      self.fmt = 'gray'
    elif len(self.shape) == 3 and self.shape[2] == 3:
      self.fmt = 'rgb8'
    else:
      raise ValueError(f"Unrecognized shape: {self.shape}.")

    self.cmd = [
      'ffmpeg',
      '-y',                     # overwrite existing file
      '-f', 'rawvideo',
      '-vcodec', 'rawvideo',                    # TODO: necessary?
      '-s', f'{self.shape[0]}x{self.shape[1]}', # frame size
      '-pix-fmt', self.fmt,                     # byte format
      '-r', str(self.fps),                      # frames per second
      '-i', '-',                                # input from pipe
      '-an',                                    # no audio
      '-vcodec', 'mpeg4',
      self.fname]

    self.proc = sp.Popen(self.cmd, stdin=sp.PIPE)
      
  def write(self, frame):
    assert frame.shape == self.shape
    proto = frame.astype(np.uint8).tostring()
    self.proc.stdin.write(proto)

  def close(self):
    self.proc.stdin.close()
    self.proc.wait()
    self.proc.close()
