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
root = "data/waltzing_spheres/"  # root dir for fname
fps = 30                         # frame rate of the video
frame_step = 1/float(fps)        # time per frame (DERIVED)
steps_per_frame = 4              # number of time steps per frame
time_step = steps_per_frame * frame_step # delta t for simulation
output_formats = {'png', 'mp4'}          # output formats
image_shape = (196, 196)                 # image shape
num_classes = 3                          # including background

# Configure initial parameters. 1 povray unit = 1 cm
# ball 1 in povray unites
r1 = 5                          # radius (cm)
x1 = 0                          # initial x position (cm)
y1 = 0                          # initial y position

# ball 2
r2 = 15
x2 = 0
y2 = 0

#################### CONFIGURABLE OPTIONS ABOVE ####################

def step(n=1):
  """

  :param n: 

  """
  global initial, current
  dt = time_step

  # Just do cartesian coordinates. Cartesian coordinates are just easier, in
  # case I have multiple things flying around.
  while (n > 0):
    for k in current.keys():
      initial[k] = current[k]

    # 1. Calculate half-step velocity
    half_step_v1 = initial['v1'] + 0.5*initial['a1'] * dt
    half_step_v2 = initial['v2'] + 0.5*initial['a2'] * dt

    # 2. Calculate current position
    current['x1'] = initial['x1'] + half_step_v1 * dt
    current['x2'] = initial['x2'] + half_step_v2 * dt

    # 3. Calculate current acceleration
    current['a1'], current['a2'] = calculate_acceleration(current['x1'],
                                                          current['x2'])

    # 4. Calculate current velocity
    current['v1'] = half_step_v1 + 0.5*current['a1'] * dt
    current['v2'] = half_step_v2 + 0.5*current['a2'] * dt

    # Correct for bouncing off of walls
    impose_walls()

    logger.debug("position:{},{}".format(current['x1'], current['x2']))
    n -= 1

    
global step_cnt
step_cnt = 0
def update_to_step(t):
  """Update to physical time step t (proportional to frame number fn)"""
  global step_cnt
  if t > step_cnt:
    if debug: print("updating to step", t)
    step(n = t - step_cnt)
    step_cnt = t

def positions():
  pass
    
# experiment spheres: whichever one is called first updates the global
# state. Then each of them translates the global state back into cartesian
# coordinates. Takes the frame number as argument.
def argsf1(fn):
  t = steps_per_frame * fn
  update_to_step(t)
  x, y = 100 * current['x1']
  return [x,y,0], r1

def argsf2(fn):
  t = steps_per_frame * fn      # TODO: fix
  update_to_step(t)
  x, y = 100 * current['x2']
  return [x,y,0], r2


def main():
  # helpers
  color = lambda col : vapory.Texture(vapory.Pigment('color', col))
  texture = lambda text : vapory.Texture(text)
  
  # initial state, in SI units
  global initial, current
  initial = {}
  initial['x1'] = np.array([x1, y1]) / 100.
  initial['v1'] = np.array([vx1, vy1]) / 100.
  initial['x2'] = np.array([x2, y2]) / 100.
  initial['v2'] = np.array([vx2, vy2]) / 100.

  # Calculate initial acceleration with equations of motion
  initial['a1'], initial['a2'] = calculate_acceleration(initial['x1'],
                                                        initial['x2'])

  current = initial.copy()

  # Begin setup
  s1 = experiment.ExperimentSphere(argsf1, texture('PinkAlabaster'),
                                   semantic_label=1)
  s2 = experiment.ExperimentSphere(argsf2, texture('PinkAlabaster'),
                                   semantic_label=2)

  # experiment
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

  if do_walls:
    global walls
    walls = np.zeros(4)         # top, left, bottom, right
    walls[:2] = exp.unproject_to_image_plane((0, 0))[:2]
    walls[2:] = exp.unproject_to_image_plane(image_shape)[:2]
    walls /= 100.               # convert to meters

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
