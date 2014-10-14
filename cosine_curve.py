from constrained_integrator import ConstrainedIntegrator
import matplotlib
from matplotlib import pyplot
import numpy as np
import sys


def PlotDistribution(path):
  theta_vector = np.linspace(0, 2*np.pi, 50)
  x = []
  y = []
  for theta in theta_vector:
    r = 1.0 + 0.25*np.cos(3.*theta)
    x.append(r*np.cos(theta))
    y.append(r*np.sin(theta))
  
  pyplot.plot(x, y, 'k-')
  
  x_path = []
  y_path = []
  for pos in path:
    # pos is a 2x1 matrix of position values.
    x_path.append(pos[0, 0])
    y_path.append(pos[1, 0])
  pyplot.plot(x_path, y_path, 'b--')
  pyplot.show()


if __name__ == "__main__":
  n_steps = int(sys.argv[1])
  dt = float(sys.argv[2])
  # Set initial condition.
  initial_position = np.matrix([[1.25], [0.0]])
  mobility = np.matrix([[1.0, 0.0], [0.0, 1.0]])
  def CurveConstraint(x):
    r = np.sqrt(x[0, 0]**2 + x[1, 0]**2)
    theta = np.arctan(x[1, 0]/x[0, 0])
    return r - 0.25*np.cos(3.*theta) - 1.

  curve_integrator = ConstrainedIntegrator(CurveConstraint, mobility,
                                           'RFD', initial_position)
  
  for k in range(n_steps):
    curve_integrator.TimeStep(dt)

  PlotDistribution(curve_integrator.path)
  
