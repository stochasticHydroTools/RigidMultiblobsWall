import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from cosine_curve import RunAnalyzer
import sys

def PlotCurve():
  theta_vector = np.linspace(0, 2*np.pi, 50)

  x = []
  y = []
  for theta in theta_vector:
    r = 1.0 + 0.25*np.cos(3.*theta)
    x.append(r*np.cos(theta))
    y.append(r*np.sin(theta))
  
  pyplot.plot(x, y, 'b--')
  pyplot.show()

if __name__ == '__main__':

  run_analyzer = RunAnalyzer(100)

  for filename in sys.argv[2:]:
    run_analyzer.load_histogram(filename)

  plot_name = sys.argv[1]
  run_analyzer.plot_theta_histogram(plot_name)
    

  
