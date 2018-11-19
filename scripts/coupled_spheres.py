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

# dataset parameters (configurable)
root = "data/coupled_spheres/"   # root dir for fname
fps = 30                         # frame rate of the video
steps_per_frame = 1              # number of simulated time steps per frame
seconds = 5                      # number of seconds in the video
N = int(seconds / time_step)     # number of frames
output_formats = {'mp4'}         # write to a video
fname = root + 'coupled_spheres' # extensions from output_formats
image_shape = (512, 512)         # image shape
num_classes = 2                  # including background

# dataset params (derived)
frame_step = 1/float(fps)                # time per frame
time_step = steps_per_frame * frame_step # time step for simulation

# initial parameters. 1 povray unit = 1 cm
initial = {}

# ball 1
r1 = 50              # radius
m1 = 2               # mass (kg)
x1 = -200            # initial x position (cm)
y1 = 0               # initial y position
vx1 = 0              # initial x velocity (cm/s)
vy1 = 0              # initial y velocity

# ball 2
r2 = 75
m2 = 3 
x2 = 200
y2 = 0
vx2 = 0
vy2 = 0

M = m1 + m2

# X = [l, l*, th, th*] is the current state of the polar-coordinate system
# Xc = [xc, xc*, yc, yc*] is the current state of the center of mass
# where '*' denotes the time derivative "dot"
global Xc
Xc = np.array([(x1*m1 + x2*m2)/M,
               (vx1*m1 + vx2*m2)/M,
               (y1*m1 + y2*m2)/M,
               (vy1*m1 + vy2*m2)/M])

xc, vxc, yc, vyc = Xc
l_sqr = (x1 - xc)**2 + (y1 - yc)**2
l = np.sqrt(l_sqr)
dl = ((x1 - xc)*(vx1 - vxc) + (y1 - yc)*(vy1 - vyc)) / l
th = np.arctan2(y1 - yc, x1 - xc)
dth = ((x1 - xc)*(vy1 - vyc) - (y1 - yc)*(vx1 - vxc)) / l_sqr

global X
X = np.array([l, dl, th, dth])

# spring:
k = 0                           # spring constant
l0 = 10                    # relaxed length (cm)
def spring(l):
  """Return the force in Newtons exerted by the spring as a function of its length
  `l`. Negative force is attractive, positive repulsive. In center-of-mass polar
  coordinates, this should be (will be) a radial force.

  In the small-displacement approximation, this should be a linear relation
  according to Hooke's law. This function allows us to encode non-linearities in
  the forces, in case I want to expand the simulation to do that.

  """
  
  return k * (l - l0)

def step(n=1, dt=time_step):
  """Update the polar and CM system over n time steps of length dt. Recall:
  l, dl, th, dth = X
  xc, vxc, yc, vyc = Xc
  """
  global X, Xc

  while (n > 0):
    ddth = - 2 * (X[0] / X[1]) * X[3]
    ddl = spring(X[0]) / M + X[0] * X[3]*X[3]
    
    dt_sqr = dt*dt
    X[0] = X[0] + X[1]*dt + 0.5*ddl*dt_sqr
    X[1] = X[1] + ddl*dt
    X[2] = X[2] + X[3]*dt + 0.5*ddth*dt_sqr
    X[3] = X[3] + ddth*dt
    n -= 1

global step_cnt
step_cnt = 0
def update_to_step(t):
  """Update to physical time step t (proportional to frame number fn)"""
  global step_cnt
  if t > step_cnt:
    step(n = t - step_cnt)
    step_cnt += t
  
# experiment spheres: whichever one is called first updates the global
# state. Then each of them translates the global state back into cartesian
# coordinates. Takes the frame number as argument.
def argsf1(fn):
  t = steps_per_frame * fn
  update_to_step(t)
  r1 = X[0] * m2 / M
  x1 = Xc[0] + r1*np.cos(X[2])
  y1 = Xc[2] + r1*np.sin(X[2])
  return [x1,y1,0], r1            # TODO: placeholder

def argsf2(fn):
  t = steps_per_frame * fn
  update_to_step(t)
  r2 = X[0] * m1 / M
  x1 = Xc[0] - r1*np.cos(X[2])
  y1 = Xc[2] - r1*np.sin(X[2])  
  return [x2,y2,0], r1            # TODO: placeholder

s1 = experiment.ExperimentSphere(argsf1, color('Gray'))
s2 = experiment.ExperimentSphere(argsf2, color('Gray'))

# experiment
exp = experiment.Experiment(image_shape=image_shape,
                            num_classes=num_classes,
                            N=N, fname=fname,
                            output_format=output_formats,
                            fps=fps, mode='L')
exp.add_object(vapory.LightSource([0, 5*image_shape[0], -5*image_shape[1]],
                                  'color', [1,1,1]))
exp.add_object(vapory.LightSource([5*image_shape[0], 0, -5*image_shape[1]],
                                  'color', [1,1,1]))
exp.add_object(vapory.Plane([0,0,1], 2*max(r1, r2), color('White'))) # ground
exp.add_object(s1)
exp.add_object(s2)

if debug:
  image, annotation = exp.render_scene(0)
  plt.imshow(image[:,:,0], cmap='gray')
  plt.show()
else:
  exp.run(verbose=True)
