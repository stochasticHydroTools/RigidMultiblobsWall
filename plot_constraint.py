import numpy as np
import matplotlib
from matplotlib import pyplot
matplotlib.use('Agg')

theta_vector = np.linspace(0, 2*np.pi, 50)

x = []
y = []
for theta in theta_vector:
  r = 1.0 + 0.25*np.cos(3.*theta)
  x.append(r*np.cos(theta))
  y.append(r*np.sin(theta))
  
pyplot.plot(x, y, 'b--')
pyplot.show()
