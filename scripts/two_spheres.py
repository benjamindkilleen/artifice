"""Create a dataset of scenes with two spheres, red and blue, of separate
classes. Vary the position and radius of these spheres randomly within the image
frame.

Outputs a tfrecord in current data/two_spheres. (Should be run from $ARTIFICE)

"""

import vapory
import numpy as np
from test_utils import experiment
from artifice.utils import dataset
import matplotlib.pyplot as plt

debug = False

# Parameters
root = "data/two_spheres/"            # Root dir for fname
N = 5 if debug else 5000               # number of examples
image_shape = (512, 512)              # first two dimensions of output images
num_classes = 2                       # number of semantic classes
fname = root + "two_spheres.tfrecord" # tfrecord to write to
min_radius = 64                       # minimum radius of either sphere
max_radius = 128                      # maximum radius of either sphere

# experiment
color = lambda col : vapory.Texture(vapory.Pigment('color', col))
exp = experiment.BallExperiment(image_shape=image_shape,
                                num_classes=num_classes,
                                N=N, fname=fname)
exp.add_object(vapory.Background('White'))
exp.add_object(vapory.LightSource([0, image_shape[0], -2*image_shape[1]],
                                  'color', [1,1,1]))
argsf = lambda : (
  [np.random.randint(-image_shape[1]/2, image_shape[1]/2),
   np.random.randint(-image_shape[0]/2, image_shape[0]/2),
   np.random.randint(-max_radius, max_radius)],
  np.random.randint(min_radius, max_radius+1))

# grayscale images, but objects will result in the same class anyway
red_ball = experiment.ExperimentSphere(argsf, color('Red'))
blue_ball = experiment.ExperimentSphere(argsf, color('Blue'))

exp.add_object(red_ball)
exp.add_object(blue_ball)

# Run the experiment, creating the tfrecord
exp.run(verbose=debug)

# Save the first two images of the data, only works in sinteractive
if debug:
  dataset.save_first_scene(fname)
