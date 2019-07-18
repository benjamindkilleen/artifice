from sys import argv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

im = Image.open(argv[1])
image = np.array(im).reshape(im.size[0], im.size[1])
plt.imshow(image, cmap='gray')
plt.show()
