'''File with utilities for the scripts and functions in this project.'''
import logging
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np


DT_STYLES = {}  # Used for plotting different timesteps of MSD.

# Fake log-like class to redirect stdout to log file.
class StreamToLogger(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   """
   def __init__(self, logger, log_level=logging.INFO):
      self.logger = logger
      self.log_level = log_level
      self.linebuf = ''
 
   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())


# Static Variable decorator for calculating acceptance rate.
def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate

class MSDStatistics(object):
  '''
  Class to hold the means and std deviations of the time dependent
  MSD for multiple schemes and timesteps.  data is a dictionary of
  dictionaries, holding runs indexed by scheme and timestep in that 
  order.
  Each run is organized as a list of 3 arrays: [time, mean, std]
  mean and std are matrices (the rotational MSD).
  '''
  def __init__(self, params):
    self.data = {}
    self.params = params


  def add_run(self, scheme_name, dt, run_data):
    ''' 
    Add a run.  Create the entry if need be. 
    run is organized as a list of 3 arrays: [time, mean, std]
    In that order.
    '''
    if scheme_name not in self.data:
      self.data[scheme_name] = dict()
    self.data[scheme_name][dt] = run_data


  def print_params(self):
     print "Parameters are: "
     print self.params
     

def plot_time_dependent_msd(msd_statistics, ind, figure, color=None,
                            label=None):
  ''' 
  Plot the <ind> entry of the rotational MSD as 
  a function of time on given figure (integer).  
  This uses the msd_statistics object
  that is saved by the *_rotational_msd.py script.
  ind contains the indices of the entry of the MSD matrix to be plotted.
  ind = [row index, column index].
  '''
  scheme_colors = {'RFD': 'g', 'FIXMAN': 'b', 'EM': 'r'}
  pyplot.figure(figure)
  # Types of lines for different dts.
  write_data = True
  if write_data:
    np.set_printoptions(threshold=np.nan)
  num_err_bars = 18
  linestyles = ['', ':', '--', '-.']
  for scheme in msd_statistics.data.keys():
    for dt in msd_statistics.data[scheme].keys():
      if dt in DT_STYLES.keys():
        dt_style = DT_STYLES[dt]
      else:
        dt_style = linestyles[len(DT_STYLES)]
        DT_STYLES[dt] = dt_style
      # Extract the entry specified by ind to plot.
      num_steps = len(msd_statistics.data[scheme][dt][0])
      # Don't put error bars at every point
      err_idx = [int(num_steps*k/num_err_bars) for k in range(num_err_bars)]
      msd_entries = np.array([msd_statistics.data[scheme][dt][1][_][ind[0]][ind[1]]
                              for _ in range(num_steps)])
      msd_entries_std = np.array(
        [msd_statistics.data[scheme][dt][2][_][ind[0]][ind[1]]
         for _ in range(num_steps)])
      # Set label and style.
      if label:
         plot_label = label
      else:
         plot_label = '%s, dt=%s' % (scheme, dt)

      if color:
         plot_style = color + dt_style
         err_bar_color = color
      else:
         plot_style = scheme_colors[scheme] + dt_style
         err_bar_color = scheme_colors[scheme]

      pyplot.plot(msd_statistics.data[scheme][dt][0],
                  msd_entries,
                  plot_style,
                  label = plot_label)
      if write_data:
        with open("./data/MSD-component-%s-%s.txt" % (ind[0], ind[1]),'w+') as f:
          f.write("scheme %s \n" % scheme)
          f.write("dt %s \n" % dt)
          f.write("time: %s \n" % msd_statistics.data[scheme][dt][0])
          f.write("MSD component: %s \n" % msd_entries)
          f.write("Std Dev:  %s \n" % msd_entries_std)
        
      pyplot.errorbar(np.array(msd_statistics.data[scheme][dt][0])[err_idx],
                      msd_entries[err_idx],
                      yerr = 2.*msd_entries_std[err_idx],
                      fmt = err_bar_color + '.')
  pyplot.ylabel('MSD')
  pyplot.xlabel('time')

