"""Create a dataset of two spheres that walk regularly along the image plane,
separated, with some step size between each one. Useful for a test-set.

Each "time step" is just a shift over. "steps_per_frame" is useful for skipping
by a few pixels at a time.

"""

import vapory
import numpy as np
import matplotlib.pyplot as plt
from test_utils import experiment
import logging

logger = logging.getLogger('experiment')

# Main parameters
debug = False

# dataset parameters
root = "data/shadowed_right_spheres/"  # root dir for fname
fps = 30                         # frame rate of the video
frame_step = 1/float(fps)        # time per frame (DERIVED)
separation = 4                   # separation between cell-centered samples
output_formats = {'png', 'mp4'}  # output formats
image_shape = (196, 196)         # image shape
num_classes = 3                  # including background
num_rows = image_shape[0] // separation
num_cols = image_shape[1] // separation
N = num_rows * num_cols

# Configure initial parameters. 1 povray unit = 1 cm
# ball 1 in povray unites
r1 = 5                          # radius (cm)

# ball 2
r2 = 15
big_sphere_offset = -image_shape[0] // 2 # N//2

#################### CONFIGURABLE OPTIONS ABOVE ####################

"""
Given the global index, which is stepping over the spots separated by
steps_per_frame pixels, calculate the index at which it would be in the image
and convert that to world-space.
"""

def compute_position(n, offset=0):
  """Return the x,y world-space position at step (n + offset) % N.

  :param n: the global step
  :param offset: offset for the sphere's starting position.
  :returns: world-space position of sphere.
  :rtype: 

  """
  global exp
  idx = (n + offset) % N
  i = separation*(idx // num_cols) + separation / 2. + 0.5
  j = separation*(idx % num_cols) + separation / 2. + 0.5
  return list(exp.unproject_to_image_plane([i,j]))[:2]
  
def argsf1(n):
  x,y = compute_position(n)
  # logger.info(f"x1: {x,y}")
  return [x,y,0], r1

def argsf2(n):
  x,y = compute_position(n, offset=big_sphere_offset)
  # logger.info(f"x2: {x,y}")
  return [x,y,0], r2


def main():
  # helpers
  color = lambda col : vapory.Texture(vapory.Pigment('color', col))
  texture = lambda text : vapory.Texture(text)
  
  # Begin setup
  s1 = experiment.ExperimentSphere(argsf1, texture('PinkAlabaster'),
                                   semantic_label=1)
  s2 = experiment.ExperimentSphere(argsf2, texture('PinkAlabaster'),
                                   semantic_label=2)

  # experiment
  global exp
  exp = experiment.Experiment(image_shape=image_shape,
                              num_classes=num_classes,
                              N=N, data_root=root,
                              output_format=output_formats,
                              fps=fps, mode='L')
  exp.add_object(vapory.LightSource([0, 5*image_shape[0], 0],
                                    'color', [1,1,1]))
  exp.add_object(vapory.LightSource([5*image_shape[0], 0, 0],
                                    'color', [1,1,1]))

  # Background
  exp.add_object(vapory.Plane([0,0,1], max(r1, r2), texture('Blue_Sky')))
  exp.add_object(s1)
  exp.add_object(s2)
  
  if debug:
    image, annotation = exp.render_scene(0)
    plt.imshow(image[:,:,0], cmap='gray')
    plt.show()
  else:
    exp.run()


if __name__ == "__main__":
  main()
