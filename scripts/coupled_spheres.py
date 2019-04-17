"""Create a dataset of two spheres coupled by an invisible spring, floating
without gravity. Outputs a tfrecord in data/coupled_spheres. (Should be run from
$ARTIFICE)

# TODO: add realistic physics simulation as in
https://gist.github.com/Zulko/f828b38421dfbee59daf, using package 'ode'.

"""

import vapory
import numpy as np
import matplotlib.pyplot as plt
from test_utils import experiment
import logging

logger = logging.getLogger('experiment')

# Main parameters
debug = False
seconds = 1 # 400                   # 12000 frames, at 30fps
tether = True                  # Tether the (large) ball to center.

# dataset parameters
root = "data/coupled_spheres{}/".format(
  "_tethered" if tether else "") # root dir for fname
fps = 30                         # frame rate of the video
frame_step = 1/float(fps)        # time per frame (DERIVED)
steps_per_frame = 5              # number of simulated time steps per frame
time_step = steps_per_frame * frame_step # delta t for simulation
N = int(fps * seconds)                   # number of frames (DERIVED)
output_formats = {'png', 'mp4'}          # output formats
image_shape = (196, 196)                 # image shape
num_classes = 3                          # including background

# Configure initial parameters. 1 povray unit = 1 cm
# ball 1 in povray unites
r1 = 10              # radius (cm)
m1 = 1               # mass (kg)
x1 = 50              # initial x position (cm)
y1 = 0               # initial y position
vx1 = -30            # initial x velocity (cm/s)
vy1 = 100            # initial y velocity

# ball 2
r2 = 30
m2 = 27
x2 = 0
y2 = 0
vx2 = 0
vy2 = 0

# Spring parameters
k = 15                          # Hooke's constant (N / m)
relaxed_length = 200            # For Hooke's law (cm)
minimum_length = r1 + r2        # Nonlinear boundary of spring (cm)

# tether parameters
tether_center = np.array(image_shape, np.float64) / 2.

# Add walls at the boundary of the image plane
do_walls = True                # TODO: fix this behavior

#################### CONFIGURABLE OPTIONS ABOVE ####################

# spring:
l_relaxed = relaxed_length / 100.
l_min = minimum_length / 100.
def spring(l):
  """
  :param l: distance between masses, in meters

  Return the force in Newtons exerted by the spring as a function of its length
  `l`. Negative force is attractive, positive repulsive. In center-of-mass polar
  coordinates, this should be (will be) a radial force.

  In the small-displacement approximation, this should be a linear relation
  according to Hooke's law. This function allows us to encode non-linearities in
  the forces, in case I want to expand the simulation to do that.

  """

  # Prevent occlusion, possibly.
  if l > l_min:
    lower_boundary = 0.1 / np.square(l - l_min) # coefficient may require tuning.
  else:
    lower_boundary = 10000      # Shouldn't happen, if above tuned correctly
  
  return -k * (l - l_relaxed) + lower_boundary

# attractor:
attractor_relaxed = 0
def attractor(l):
  """Return a spring-like force as a function of mag_l

  :param l: distance from object to attractor, in meters
  :returns: attractive force in Newtons

  """
  return -k * (l - attractor_relaxed)


def calculate_acceleration(x1, x2):
  """Calculate the accelerations of the system from equations of motion, given
  position vectors x1 and x2 for the two spheres.
  """
  l = x1 - x2
  mag_l = np.linalg.norm(l)
  mag_F = spring(mag_l)
  l_hat = l / mag_l
  a1 = mag_F * l_hat / m1
  a2 = -mag_F * l_hat / m2
  if tether:
    l = x2 - tether_center
    mag_l = np.linalg.norm(l)
    mag_F = attractor(mag_l)
    l_hat = l / mag_l
    a2 -= mag_F
  return a1, a2

def impose_walls():
  """Impose the walls at the boundary of the image_plane on the CURRENT state of
  the system.
  `walls` consists of top, left, bottom, right bounds.
  """
  if not do_walls:
    return

  global current, walls
  for objNum in ['1', '2']:
    xk = 'x' + objNum
    vk = 'v' + objNum
    if current[xk][0] < walls[1]:
      logger.debug(f"bouncing off left wall at {walls[1]}")
      current[xk][0] = 2*walls[1] - current[xk][0]
      current[vk][0] *= -1
    if current[xk][0] > walls[3]:
      logger.debug(f"bouncing off right wall at {walls[3]}")
      current[xk][0] = 2*walls[3] - current[xk][0]
      current[vk][0] *= -1
    if current[xk][1] > walls[0]:
      logger.debug(f"bouncing off top wall at {walls[0]}")
      current[xk][1] = 2*walls[0] - current[xk][1]
      current[vk][1] *= -1
    if current[xk][1] < walls[2]:
      logger.debug(f"bouncing off bottom wall at {walls[2]}")
      current[xk][1] = 2*walls[2] - current[xk][1]
      current[vk][1] *= -1


def step(n=1):
  """Update the polar and CM system over n time steps of length dt, using the
  velocity Verlet algorithm, as on
  https://en.wikipedia.org/wiki/Verlet_integration

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
