"""Create a (test) dataset of a single sphere flying in a parabolic arc.
Outputs a tfrecord in current data/two_spheres. (Should be run from $ARTIFICE)
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
root = "data/arcing_sphere/"    # root dir for fname
time_step = 1/30.               # time per frame
seconds = 5                     # number of seconds in the video
N = int(seconds / time_step)    # number of frames
output_formats = {'mp4'}        # write to a video
fname = 'sphere_arcing'         # extensions from output_formats
image_shape = (512, 512)        # image shape
num_classes = 2                 # including background

# physical sphere parameters
radius = 50                     # radius in approx-pixels. (e.g. meters)
mass = 2                        # mass in kilograms
x = -iamge_shape[0] / 2         # initial x position in world
y = -300                        # initial y position in world
vx = 10                         # initial x velocity in m/s
vy = 10                         # initial y velocity in m/s
g = -9.81                       # gravity acceleration, in m/s/s

# experiment sphere parameters
def argsf(t_):
  t = t_ * time_step
  return ([vx*t + x, 0.5*g*t**2 + vy*t + y, 0], radius)

ball = experiment.ExperimentSphere(argsf, color('Blue'))

# experiment
exp = experiment.Experiment(image_shape=image_shape,
                            num_classes=num_classes,
                            N=N, fname=fname,
                            output_format=output_formats)
exp.add_object(vapory.LightSource([0, image_shape[0], -2*image_shape[1]],
                                  'color', [1,1,1]))
exp.add_object(vapory.Plane([0,1,0], y - radius, color('White'))) # ground
exp.add_object(vapory.Plane([0,0,1], -5*radius, color('White')))  # back wall
exp.add_object(ball)

exp.run(verbose=debug)

