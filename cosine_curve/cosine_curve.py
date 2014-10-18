'''  
Script to run constrained diffusion on the curve 
  r = 1 + 0.25*cos(3*theta)
in two dimensions.  Currently uses no potential, and uses mobility
diag(1 + 0.25*x_i^2).
run with:
python cosine_curve.py dt nsteps nruns
to run nruns trajectories of nsteps each with timestep dt.
'''

from constrained_integrator import ConstrainedIntegrator
import matplotlib
from matplotlib import pyplot
import numpy as np
import sys
import cProfile, pstats, StringIO
import cosine_curve_ext  # Functions implemented in C++ for speed.
import cPickle

PROFILE = 0

class RunAnalyzer:
  ''' Small class to store historgrams from runs and plot the results. '''
  def __init__(self, resolution):
    self.theta_hists = []
    self.resolution = resolution
    self.bins = np.linspace(-np.pi/2.,3.*np.pi/2.,self.resolution)

  def PlotDistribution(self, path):
    ''' plot the path '''
    theta_vector = np.linspace(0, 2*np.pi, self.resolution)
    x = []
    y = []
    for theta in theta_vector:
      r = 1.0 + 0.25*np.cos(3.*theta)
      x.append(r*np.cos(theta))
      y.append(r*np.sin(theta))
  
    pyplot.plot(x, y, 'k-')
    pyplot.plot([pos[0, 0] for pos in path], [pos[1, 0] for pos in path], 'b--')
    pyplot.show()

  def BinTheta(self, path):
    ''' Bin the thetas from this particular run '''
    theta_path = []
    for pos in path:
      # pos is a 2x1 matrix of position values.
      theta = np.arctan(pos[1, 0]/pos[0, 0]) + (np.pi)*(pos[0, 0] < 0)
      theta_path.append(theta)

    hist = np.histogram(theta_path, bins=self.bins,
                        density=True)

    self.theta_hists.append(hist[0])

  
  def SaveHistogram(self, filename):
    ''' 
    Save the histogram as a pkl object to combine with other runs and 
    plot. 
    args:
       filename:  string - name of the file to save to.
    '''
    with open('./data/' + filename,'wb') as f:
      cPickle.dump(self.theta_hists, f)

    
  def LoadHistogram(self, filename):
    '''
    Load histograms from a given file and append to 
    self.theta_hists
    '''
    with open('./data/' + filename, 'rb') as f:
      loaded_theta_hists = cPickle.load(f)
    for hist in loaded_theta_hists:
      self.theta_hists.append(hist)

      
  def PlotThetaHistogram(self, filename):
    ''' plot the mean and std def of all path binned with BinTheta '''

    bin_centers = (self.bins[:-1] + self.bins[1:])/2.
    n_runs = len(self.theta_hists)
    print n_runs
    theta_means = []
    theta_stds = []
    for k in range(self.resolution-1):
      runs_at_this_theta = [self.theta_hists[j][k] for j in range(n_runs)]
      theta_means.append(np.mean(runs_at_this_theta))
      theta_stds.append(np.std(runs_at_this_theta)/np.sqrt(n_runs))
      
    theory_dist = (1. + 0.25*np.cos(3.*bin_centers))/(2.*np.pi)
    pyplot.plot(bin_centers, theory_dist, 'k--')
    pyplot.errorbar(bin_centers, theta_means, yerr=2.*np.array(theta_stds))
    pyplot.savefig("./Plots/" + str(filename))
    

if __name__ == "__main__":
  if PROFILE:
    pr = cProfile.Profile()
    pr.enable()

  dt = float(sys.argv[1])
  n_steps = int(sys.argv[2])
  n_runs = int(sys.argv[3])
  data_name = sys.argv[4]
  plot_name = './ThetaDistribution-dt-%s-n-%s-runs-%s.pdf' % (dt, n_steps, n_runs)
  # Set initial condition.
  initial_position = np.matrix([[1.25], [0.0]])
  def MobilityFunction(x):
    return np.matrix([[1. + 0.25*x[0, 0]**2, 0.], 
                      [0., 1. + 0.25*x[1, 0]**2]])

  def CurveConstraint(x):
    return cosine_curve_ext.CosineConstraint(x[0, 0], x[1, 0])

  curve_integrator = ConstrainedIntegrator(CurveConstraint, MobilityFunction,
                                           'RFD', initial_position)

  # Argument is the resolution for the bins.
  run_analyzer = RunAnalyzer(100)

  for j in range(n_runs):
    for k in range(n_steps):
      curve_integrator.TimeStep(dt)
    run_analyzer.BinTheta(curve_integrator.path)
    print "Completed run ", j

  run_analyzer.SaveHistogram(data_name)
  
  if PROFILE:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
