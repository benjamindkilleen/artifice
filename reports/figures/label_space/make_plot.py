import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import stringcase

def pos_outside_ellipse(pos, ell):
  """Check if pos is inside the given ellipse."""
  x,y = pos
  ex, ey = ell.center
  ea, eb = ell.width / 2, ell.height / 2
  return (x - ex)**2 / ea**2 + (y - ey)**2 / eb**2 > 1.

def outside_ellipse(positions, ell):
  out = np.zeros(positions.shape[0], dtype=bool)
  for i, pos in enumerate(positions):
    out[i] = pos_outside_ellipse(pos, ell)
  return out

def draw_inside_ellipse(ell, dist=np.random.uniform, **kwargs):
  """Draw size points from a distribution inside the ellipse"""
  points = dist(**kwargs)
  outside = outside_ellipse(points, ell)
  while outside.any():
    new_points = dist(**kwargs)
    points[outside] = new_points[outside]
    outside = outside_ellipse(points, ell)
  return points

def uniform_ellipse(ell, size):
  ex, ey = ell.center
  ea, eb = ell.width / 2, ell.height / 2
  low = (ex - ea, ey - eb)
  high = (ex + ea, ey + eb)
  return draw_inside_ellipse(ell, low=low, high=high, size=(size,2))

def make_ellipse(pos, sx, sy):
  ell = Ellipse(pos, 7.2*sx, 7.2*sy)
  return ell

def add_ellipse(ell, color='b'):
  ax = plt.gca()
  ell_ = Ellipse(ell.center, ell.width, ell.height, fill=False, linestyle='-',
                 linewidth=2., rasterized=False, edgecolor=color)
  ax.add_artist(ell_)

mean1 = [150, 194]
sx1 = sy1 = 20

mean2 = [280, 194]
sx2 = 20
sy2 = 50
sl=0

size=800
markersize=2

ell_1 = make_ellipse(mean1, sx2, sy2)  
ell_2 = make_ellipse(mean2, sx2, sy2)
x1,y1 = draw_inside_ellipse(ell_1, dist=np.random.multivariate_normal,
                            mean=mean1, cov=[[sx1*sx1,0], [0,sy1*sy1]],
                            size=size).T
x2,y2 = draw_inside_ellipse(ell_2, dist=np.random.multivariate_normal,
                            mean=mean2, cov=[[sx2*sx2,sl*sl], [sl*sl,sy2*sy2]],
                            size=size).T

def save_plot(title, ext='pdf'):
  fig = plt.gcf()
  fig.set_size_inches(5, 5)
  plt.axis('equal')
  plt.axis([0,388, 388, 0])
  # plt.xlim(0,388)
  # plt.ylim(388,0)
  title = title.lower().replace(' ', '_')
  plt.title(stringcase.titlecase(title))
  fname = stringcase.snakecase(title)
  plt.savefig(f'../images/{fname}.{ext}')
  plt.close()
  
def make_original():
  plt.plot(x1,y1, 'b.', markersize=markersize)
  plt.plot(x2,y2, 'g.', markersize=markersize)
  save_plot('Object Positions in Label Space')

def make_boundaries():
  plt.plot(x1,y1, 'b.', markersize=markersize)
  plt.plot(x2,y2, 'g.', markersize=markersize)
  add_ellipse(ell_1, 'b')
  add_ellipse(ell_2, 'g')
  save_plot('Known Label Space Boundaries')

def make_bad_sampling():
  plt.plot(x1,y1, 'b.', markersize=markersize)
  plt.plot(x2,y2, 'g.', markersize=markersize)
  add_ellipse(ell_1, 'b')
  add_ellipse(ell_2, 'g')
  ex1, ey1 = uniform_ellipse(ell_1, 3*size).T
  ex2, ey2 = uniform_ellipse(ell_2, 3*size).T
  plt.plot(ex1, ey1, 'b.', markersize=markersize)
  plt.plot(ex2, ey2, 'g.', markersize=markersize)
  save_plot('Augmentation without Resampling')
  
def make_uniform():
  add_ellipse(ell_1, 'b')
  add_ellipse(ell_2, 'g')
  ex1, ey1 = uniform_ellipse(ell_1, 3*size).T
  ex2, ey2 = uniform_ellipse(ell_2, 3*size).T
  plt.plot(ex1, ey1, 'b.', markersize=markersize)
  plt.plot(ex2, ey2, 'g.', markersize=markersize)
  save_plot('Resampling from Known Label Space Boundary')

make_original()
make_boundaries()
make_bad_sampling()
make_uniform()
