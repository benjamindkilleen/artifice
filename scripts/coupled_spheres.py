"""Create a dataset of two spheres coupled by an invisible spring, floating
without gravity. Outputs a tfrecord in data/coupled_spheres. (Should be run from
$ARTIFICE)

# TODO: add gravitational/EM attraction to center?

"""

import vapory
import numpy as np
import matplotlib.pyplot as plt

from test_utils import experiment
from artifice.utils import dataset

debug = True

# helpers
color = lambda col : vapory.Texture(vapory.Pigment('color', col))

# dataset parameters
root = "data/coupled_spheres/"   # root dir for fname
fps = 30                         # frame rate of the video
time_step = 1/float(fps)         # time per frame
seconds = 5                      # number of seconds in the video
N = int(seconds / time_step)     # number of frames
output_formats = {'mp4'}         # write to a video
fname = root + 'coupled_spheres' # extensions from output_formats
image_shape = (512, 512)         # image shape
num_classes = 2                  # including background

# physical sphere parameters. 1 povray unit = 1 cm
# ball 1
r1 = 50                         # radius
m1 = 2                          # mass (kg)
x1 = -200                       # initial x position (cm)
y1 = 0                          # initial y position
vx1 = 0                         # initial x velocity (cm/s)
vy1 = 0                         # initial y velocity

# ball 2
r2 = 75
m2 = 3 
x2 = 200
y2 = 0
vx2 = 0
vy2 = 0

# spring:
k = 0                           # spring constant
l = 10                          # relaxed length (cm)
def spring(x):
  """Return the force exerted by the spring as a function of its length. Negative
  force is attractive, positive repulsive. In center-of-mass polar coordinates,
  this should be (will be) a radial force.

  """
  return 0

# experiment sphere parameters
def argsf(t_):
  t = t_ * time_step
  return [0,0,0], r1            # TODO: placeholder

ball = experiment.ExperimentSphere(argsf, color('Red'))

# experiment
exp = experiment.Experiment(image_shape=image_shape,
                            num_classes=num_classes,
                            N=N, fname=fname,
                            output_format=output_formats,
                            fps=fps, mode='L')
exp.add_object(vapory.LightSource([0, 5*image_shape[0], -5*image_shape[1]],
                                  'color', [1,1,1]))
exp.add_object(vapory.Plane([0,1,0], y - radius, color('White'))) # ground
exp.add_object(vapory.Plane([0,0,1], 5*radius, color('White')))  # back wall
exp.add_object(ball)

if debug:
  image, annotation = exp.render_scene(0)
  plt.imshow(image[:,:,0], cmap='gray')
  plt.show()
else:
  exp.run(verbose=True)
