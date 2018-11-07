"""Create a dataset of scenes with two spheres, red and blue, of separate
classes. Vary the position and radius of these spheres randomly within the image
frame.

Outputs a tfrecord in 

"""
import vapory
import numpy as np
from test_utils import experiment


# Parameters
image_shape = (512, 512)        # first two dimensions of output images
num_classes = 2                 # number of semantic classes
N = 1                           # number of examples
fname = "two_spheres.tfrecord"  # tfrecord to write to
min_radius = 8
max_radius = 128

color = lambda col : vapory.Texture(vapory.Pigment('color', col))
exp = experiment.BallExperiment(image_shape=image_shape,
                                num_classes=num_classes,
                                N=N, fname=fname)
exp.add_object(vapory.Background('Black'))
exp.add_object(vapory.LightSource([0, 5*image_shape[0], 5*image_shape[1]],
                                  'color', [1,1,1]))
argsf = lambda : (
  [np.random.randint(-image_shape[1], image_shape[1]),
   np.random.randint(-image_shape[0], image_shape[0]),
   np.random.randint(-image_shape[0]/2, image_shape[0]/2)],
  np.random.randint(min_radius, max_radius+1))

red_ball = experiment.ExperimentSphere(argsf, color('Red'))
blue_ball = experiment.ExperimentSphere(argsf, color('Blue'))

exp.run()


