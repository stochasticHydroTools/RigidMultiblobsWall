from constrained_integrator import ConstrainedIntegrator
import matplotlib
from matplotlib import pyplot
import numpy as np
import sys
import cProfile, pstats, StringIO

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


def PlotThetaHistogram(path):
  theta_path = []
  for pos in path:
    # pos is a 2x1 matrix of position values.
    theta = np.arctan(pos[1, 0]/pos[0, 0]) + (np.pi)*(pos[0, 0] < 0)
    theta_path.append(theta)
  
  hist = np.histogram(theta_path, bins=np.linspace(-np.pi/2.,3.*np.pi/2.,100),
                      density=True)

  bin_centers = (hist[1][:-1] + hist[1][1:])/2.
  pyplot.plot(bin_centers, hist[0])
  
  theory_dist = (1. + 0.25*np.cos(3.*bin_centers))/(2.*np.pi)
  pyplot.plot(bin_centers, theory_dist, 'k--')
  pyplot.show()
    

if __name__ == "__main__":
  PROFILE = 0
  if PROFILE:
    pr = cProfile.Profile()
    pr.enable()

  n_steps = int(sys.argv[1])
  dt = float(sys.argv[2])
  # Set initial condition.
  initial_position = np.matrix([[1.25], [0.0]])
  def MobilityFunction(x):
    return np.matrix([[1. + 0.25*x[0, 0]**2, 0.], 
                      [0., 1. + 0.25*x[1, 0]**2]])

  def CurveConstraint(x):
    r_squared = (x[0, 0]**2 + x[1, 0]**2)
    theta = np.arctan(x[1, 0]/x[0, 0]) + (np.pi)*(x[0, 0] < 0)
    return r_squared - (0.25*np.cos(3.*theta) + 1.)**2

  curve_integrator = ConstrainedIntegrator(CurveConstraint, MobilityFunction,
                                           'RFD', initial_position)
  
  for k in range(n_steps):
    curve_integrator.TimeStep(dt)

  PlotDistribution(curve_integrator.path)
  PlotThetaHistogram(curve_integrator.path)

  if PROFILE:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
