import numpy as np
from PIL import Image
from skimage.draw import circle, polygon

def dot_mark():
  img = np.zeros((200, 200, 4), dtype=np.uint8)

  # draw the circle around the full image
  rr, cc = circle(100, 100, 100, shape=img.shape)
  img[rr,cc,:] = np.array([0, 0, 0, 255], dtype=np.uint8)

  # hollow out the center with white
  rr, cc = circle(100, 100, 90, shape=img.shape)
  img[rr,cc,:] = np.array([255, 255, 255, 255], dtype=np.uint8)
  
  # center circle
  rr, cc = circle(100, 100, 10, shape=img.shape)
  img[rr,cc,:] = np.array([0, 0, 0, 255], dtype=np.uint8)

  return img

def noise_mark():
  img = np.zeros((200, 200, 4), dtype=np.uint8)

  # draw the circle around the full image
  rr, cc = circle(100, 100, 100, shape=img.shape)
  img[rr,cc,:] = np.array([0, 0, 0, 255], dtype=np.uint8)

  # hollow out the center with noise
  rr, cc = circle(100, 100, 90, shape=img.shape)
  noise_array = (255*np.random.normal(0.5, 0.2, size=(200, 200))).astype(np.uint8)
  img[rr,cc,0] = img[rr,cc,1] = img[rr,cc,2] = noise_array[rr,cc]
  img[rr,cc,3] = 255
  
  # center circle
  rr, cc = circle(100, 100, 10, shape=img.shape)
  img[rr,cc,:] = np.array([0, 0, 0, 255], dtype=np.uint8)

  return img

def plus_mark():
  img = np.zeros((200, 200, 4), dtype=np.uint8)

  # draw the circle around the full image
  rr, cc = circle(100, 100, 100, shape=img.shape)
  img[rr, cc, :] = np.array([0, 0, 0, 255], dtype=np.uint8)

  # hollow out the inside of the circle
  rr, cc = circle(100, 100, 90, shape=img.shape)
  img[rr,cc,:] = np.array([255, 255, 255, 255], dtype=np.uint8)

  # vertical line
  rr, cc = polygon([95, 105, 105, 95],
                   [0, 0, 200, 200],
                   shape=img.shape)
  img[rr, cc, :] = np.array([0, 0, 0, 255], dtype=np.uint8)

  # horizontal line
  rr, cc = polygon([0, 0, 200, 200],
                   [95, 105, 105, 95],
                   shape=img.shape)
  img[rr, cc, :] = np.array([0, 0, 0, 255], dtype=np.uint8)

  return img

def quarter_mark():
  img = np.zeros((200, 200, 4), dtype=np.uint8)

  # draw the circle around the full image
  rr, cc = circle(100, 100, 100, shape=img.shape)
  img[rr,cc,:] = np.array([0, 0, 0, 255], dtype=np.uint8)

  # hollow out the center, make the gray quarter circle
  rr, cc = circle(100, 100, 90, shape=img.shape)
  img[rr,cc,:] = np.array([255, 255, 255, 255], dtype=np.uint8)
  sl = np.array([rr[i] < 100 and cc[i] > 100 for i in range(rr.shape[0])])
  img[rr[sl],cc[sl],:] = np.array([127, 127, 127, 255], dtype=np.uint8)

  # vertical line
  rr, cc = polygon([95, 105, 105, 95],
                   [0, 0, 200, 200],
                   shape=img.shape)
  img[rr, cc, :] = np.array([0, 0, 0, 255], dtype=np.uint8)

  # horizontal line
  rr, cc = polygon([0, 0, 200, 200],
                   [95, 105, 105, 95],
                   shape=img.shape)
  img[rr, cc, :] = np.array([0, 0, 0, 255], dtype=np.uint8)

  return img

Image.fromarray(dot_mark()).save("dot_mark.png")
Image.fromarray(noise_mark()).save("noise_mark.png")
Image.fromarray(plus_mark()).save("plus_mark.png")
Image.fromarray(quarter_mark()).save("quarter_mark.png")

