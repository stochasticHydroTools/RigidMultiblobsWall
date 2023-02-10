'''File with utilities for the scripts and functions in this project.'''


import logging
try:
  import matplotlib
  matplotlib.use('Agg')
  from matplotlib import pyplot
except ImportError:
  pass
import numpy as np
import scipy.sparse.linalg as scspla
import os
import sys
import time
from functools import partial
from quaternion_integrator.quaternion import Quaternion 


# from quaternion_integrator.quaternion import Quaternion

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


class Tee(object):
  def __init__(self, *files):
    self.files = files
  def write(self, obj):
    for f in self.files:
      f.write(obj)
      f.flush() # If you want the output to be visible immediately
  def flush(self) :
    for f in self.files:
      f.flush()

# Static Variable decorator for calculating acceptance rate.
def static_var(varname, value):
  def decorate(func):
    setattr(func, varname, value)
    return func
  return decorate

def set_up_logger(file_name):
  ''' Set up logging info, write all print statements and errors to
  file_name.'''
  progress_logger = logging.getLogger('Progress Logger')
  progress_logger.setLevel(logging.INFO)
  # Add the log message handler to the logger
  logging.basicConfig(filename=file_name,
                      level=logging.INFO,
                      filemode='w')
  sl = StreamToLogger(progress_logger, logging.INFO)
  sys.stdout = sl
  sl = StreamToLogger(progress_logger, logging.ERROR)
  sys.stderr = sl
  return progress_logger


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
     print("Parameters are: ")
     print(self.params)
     

def plot_time_dependent_msd(msd_statistics, ind, figure, color=None, symbol=None,
                            label=None, error_indices=[0, 1, 2, 3, 4, 5], data_name=None,
                            num_err_bars=None):
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
  data_write_type = 'w'
  if not data_name:
     data_name = "MSD-component-%s-%s.txt" % (ind[0], ind[1])
  if write_data:
    np.set_printoptions(threshold=np.nan)
    with open(os.path.join('.', 'data', data_name), data_write_type) as f:
      f.write('  \n')
  if not num_err_bars:
     num_err_bars = 40
  linestyles = [':', '--', '-.', '']
  for scheme in list(msd_statistics.data.keys()):
    for dt in list(msd_statistics.data[scheme].keys()):
      if dt in list(DT_STYLES.keys()):
        if not symbol:
          style = ''
          nosymbol_style = DT_STYLES[dt]
        else:
          style = symbol #+ DT_STYLES[dt]
          nosymbol_style = DT_STYLES[dt]
      else:
        if not symbol:
          style = '' #linestyles[len(DT_STYLES)]
          DT_STYLES[dt] = linestyles[len(DT_STYLES)]
          nosymbol_style = DT_STYLES[dt]
        else:
          DT_STYLES[dt] = linestyles[len(DT_STYLES)]
          style = symbol #+ DT_STYLES[dt]
          nosymbol_style = DT_STYLES[dt]
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
        if scheme == 'FIXMAN':
          plot_label = scheme.capitalize() + label
        else:
          plot_label = scheme + label
      else:
        plot_label = '%s, dt=%s' % (scheme, dt)

      if color:
        plot_style = color + style
        nosymbol_plot_style = color + nosymbol_style
        err_bar_color = color
      else:
        plot_style = scheme_colors[scheme] + style
        nosymbol_plot_style = scheme_colors[scheme] + nosymbol_style
        err_bar_color = scheme_colors[scheme]

      pyplot.plot(np.array(msd_statistics.data[scheme][dt][0])[err_idx],
                  msd_entries[err_idx],
                  plot_style,
                  label = plot_label)
      pyplot.plot(msd_statistics.data[scheme][dt][0],
                  msd_entries,
                  nosymbol_plot_style)
      
      if write_data:
        with open(os.path.join('.', 'data', data_name),'a') as f:
          f.write("scheme %s \n" % scheme)
          f.write("dt %s \n" % dt)
          f.write("time: %s \n" % msd_statistics.data[scheme][dt][0])
          f.write("MSD component: %s \n" % msd_entries)
          f.write("Std Dev:  %s \n" % msd_entries_std)

      if ind[0] in error_indices:
        pyplot.errorbar(np.array(msd_statistics.data[scheme][dt][0])[err_idx],
                        msd_entries[err_idx],
                        yerr = 2.*msd_entries_std[err_idx],
                        fmt = err_bar_color + '.')
  pyplot.ylabel('MSD')
  pyplot.xlabel('time')

def log_time_progress(elapsed_time, time_units, total_time_units):
  ''' Write elapsed time and expected duration to progress log.'''
  progress_logger = logging.getLogger('progress_logger')  
  if elapsed_time > 60.0:
    progress_logger.info('Elapsed Time: %.2f Minutes.' % 
                         (float(elapsed_time/60.)))
  else:
    progress_logger.info('Elapsed Time: %.2f Seconds' % float(elapsed_time))

  # Report estimated duration.
  if time_units > 0:
    expected_duration = elapsed_time*total_time_units/time_units
    if expected_duration > 60.0:
      progress_logger.info('Expected Duration: %.2f Minutes.' % 
                           (float(expected_duration/60.)))
    else:
      progress_logger.info('Expected Duration: %.2f Seconds' % 
                           float(expected_duration))
      

def calc_total_msd_from_matrix_and_center(original_center, original_rotated_e, 
                                          final_center, rotated_e):
  ''' 
  Calculate 6x6 MSD including orientation and location.  This is
  calculated from precomputed center of the tetrahedron and rotation
  matrix data to avoid repeating computation.
  '''
  du_hat = np.zeros(3)
  for i in range(3):
    du_hat += 0.5*np.cross(original_rotated_e[i],
                           rotated_e[i])
    
  dx = np.array(final_center) - np.array(original_center)
  displacement = np.concatenate([dx, du_hat])
  return np.outer(displacement, displacement)


def calc_msd_data_from_trajectory(locations, orientations, calc_center_function, dt, end,
                                  burn_in = 0, trajectory_length = 100):
  ''' Calculate rotational and translational (6x6) MSD matrix given a dictionary of
  trajectory data.  Return a numpy array of 6x6 MSD matrices, one for each time.
  params:
    locations: a list of length 3 lists, indication location of the rigid body
               at each timestep.
    orientations: a list of length 4 lists, indication entries of a quaternion
               representing orientation of the rigid body at each timestep.

    calc_center_function: a function that given location and orientation
                 (as a quaternion) computes the center of the body (or the point
                 that we use to track location MSD).

    dt:  timestep used in this simulation.
    end:  end time to which we calculate MSD.
    burn_in: how many steps to skip before calculating MSD.  This is 0 by default
          because we assume that the simulation starts from a sample from the 
          Gibbs Boltzman distribution.
    trajectory_length:  How many points to keep in the window 0 to end.
              The code will process every n steps to make the total 
              number of analyzed points roughly this value.
 '''
  data_interval = int(end/dt/trajectory_length) + 1
  print("data_interval is ", data_interval)
  n_steps = len(locations)
  e_1 = np.array([1., 0., 0.])
  e_2 = np.array([0., 1., 0.])
  e_3 = np.array([0., 0., 1.])

  if trajectory_length*data_interval > n_steps:
    raise Exception('Trajectory length is longer than the total run. '
                    'Perform a longer run, or choose a shorter end time.')
  print_increment = int(n_steps/20)
  average_rotational_msd = np.array([np.zeros((6, 6)) 
                                     for _ in range(trajectory_length)])
  lagged_rotation_trajectory = []
  lagged_location_trajectory = []
  start_time = time.time()
  for k in range(n_steps):
    if k > burn_in and (k % data_interval == 0):
      orientation = Quaternion(orientations[k])
      R = orientation.rotation_matrix()
      u_hat = [np.inner(R, e_1),
               np.inner(R, e_2),
               np.inner(R,e_3)]
      lagged_rotation_trajectory.append(u_hat)
      lagged_location_trajectory.append(calc_center_function(locations[k], orientation))
    if len(lagged_location_trajectory) > trajectory_length:
      lagged_location_trajectory = lagged_location_trajectory[1:]
      lagged_rotation_trajectory = lagged_rotation_trajectory[1:]
      for l in range(trajectory_length):
        current_rot_msd = (calc_total_msd_from_matrix_and_center(
          lagged_location_trajectory[0],
          lagged_rotation_trajectory[0],
          lagged_location_trajectory[l],
          lagged_rotation_trajectory[l]))
        average_rotational_msd[l] += current_rot_msd
    if (k % print_increment) == 0 and k > 0:
      print('At step %s of %s' % (k, n_steps))
      print('For this run, time status is:')
      elapsed = time.time() - start_time
      log_time_progress(elapsed, k, n_steps)

  average_rotational_msd = (average_rotational_msd/
                            (n_steps/data_interval - trajectory_length - 
                             burn_in/data_interval))
  
  return average_rotational_msd
   

def fft_msd(x, y, end):
  ''' Calculate scalar MSD between x and yusing FFT. 
  We want D(tau) = sum(  (x(t+tau) -x(t))*(y(t+tau) - y(t)) )
  This is computed with

  D(tau) = sum(x(t)y(t)) + sum(x(t+tau)y(t+tau) - sum(x(t)*x(t+tau)
           - sum(y(t)x(t+tau))
 
  Where the last 2 sums are performed using an FFT.
  We expect that x and y are the same length.

  WARNING: THIS IS NOT CURRENTLY USED OR TESTED THOROUGHLY'''

  if len(x) != len(y):
    raise Exception('Length of X and Y are not the same, aborting MSD '
                    'FFT calculation.')
  xy_sum_tau = np.cumsum(x[::-1]*y[::-1])[::-1]/np.arange(len(x), 0, -1)
  xy_sum_t   = np.cumsum(x*y)[::-1]/np.arange(len(x), 0, -1)

  x_fft = np.fft.fft(x)
  y_fft = np.fft.fft(y)
  x_fft_xy = np.zeros(len(x))
  x_fft_yx = np.zeros(len(x))
  x_fft_xy[1:] = (x_fft[1:])*(y_fft[:0:-1])
  x_fft_xy[0] = x_fft[0]*y_fft[0]
  x_fft_yx[1:] = (y_fft[1:])*(x_fft[:0:-1])
  x_fft_yx[0] = x_fft[0]*y_fft[0]
  x_ifft_xy = np.fft.ifft(x_fft_xy)/np.arange(len(x), 0, -1)
  x_ifft_yx = np.fft.ifft(x_fft_yx)/np.arange(len(x), 0, -1)
  
  return (xy_sum_tau + xy_sum_t - x_ifft_yx - x_ifft_xy)[:end]


def write_trajectory_to_txt(file_name, trajectory, params, location=True):
  '''  
  Write parameters and data to a text file. Parameters first, then the trajectory
  one step at a time.
  '''
  # First check that the directory exists.  If not, create it.
  dir_name = os.path.dirname(file_name)
  if not os.path.isdir(dir_name):
     os.mkdir(dir_name)

  # Write data to file, parameters first then trajectory.
  with open(file_name, 'w') as f:
    f.write('Parameters:\n')
    for key, value in list(params.items()):
      f.writelines(['%s: %s \n' % (key, value)])
    f.write('Trajectory data:\n')
    if location:
      f.write('Location, Orientation:\n')
      for k in range(len(trajectory[0])):
        x = trajectory[0][k]
        theta = trajectory[1][k]
        f.write('%s, %s, %s, %s, %s, %s, %s \n' % (
          x[0], x[1], x[2], theta[0], theta[1], theta[2], theta[3]))
    else:
      f.write('Orientation:\n')
      for k in range(len(trajectory[0])):
        theta = trajectory[0][k]
        f.write('%s, %s, %s, %s \n' % (
          theta[0], theta[1], theta[2], theta[3]))

def read_trajectory_from_txt(file_name, location=True):
  ''' 
  Read a trajectory and parameters from a text file.
  '''
  params = {}
  locations = []
  orientations = []
  with open(file_name, 'r') as f:
    # First line should be "Parameters:"
    line = f.readline()
    line = f.readline()
    while line != 'Trajectory data:\n':
      items = line.split(':')
      if items[1].strip()[0] == '[':
        last_token = items[1].strip()[-1]
        list_items = items[1].strip()[1:].split('  ')
        params[items[0]] = list_items
        while last_token != ']':
          line = f.readline()
          list_items = line.strip().split('  ')
          last_token = list_items[-1].strip()[-1]
          if last_token == ']':
            list_items[-1]  = list_items[-1].strip()[:-1]
          params[items[0]] += list_items
      else:
        params[items[0]] = items[1]
      line = f.readline()
    # Read next line after 'Trajectory data' 'Location, Orientation'
    line = f.readline()
    line = f.readline()
    if location:
      while line != '':
        loc = line.split(',')
        locations.append([float(x) for x in loc[0:3]])
        orientations.append([float(x) for x in loc[3:7]])
        line = f.readline()
        
    else:
      # These two lines are '\n', and 'Orientation' 
      line = f.readline()
      line = f.readline()
      while line != '':
        quaternion_entries = line.split(',')
        orientations.append(Quaternion([float(x) for x in quaternion_entries]))
        line = f.readline()
      
  return params, locations, orientations


def transfer_mobility(mobility_1, point_1, point_2):
  '''
  Calculate mobility at point 2 based on mobility
  at point_1 of the body.  This calculates the entire 
  force and torque mobility. 
  args:
    mobility_1:  mobility matrix (force, torque) -> 
            (velocity, angular velocity) evaluated at point_1.
    point_1:  3 dimensional point where mobility_1 is evaluated.
    point_2:  3 dimensional point where we want to know the mobility
  returns:
    mobility_2: The mobility matrix evaluated at point_2.

   This uses formula (10) and (11) from:
  "Bernal, De La Torre - Transport Properties and Hydrodynamic Centers 
   of Rigid Macromolecules with Arbitrary Shapes"
  '''
  r = np.array(point_1) - np.array(point_2)
  mobility_2 = np.zeros([6, 6])
  # Rotational mobility is the same.
  mobility_2[3:6, 3:6] = mobility_1[3:6, 3:6]
  
  mobility_2[3:6, 0:3] = mobility_1[3:6, 0:3]
  mobility_2[3:6, 0:3] += tensor_cross_vector(mobility_1[3:6, 3:6], r)

  mobility_2[0:3, 3:6] = mobility_2[3:6, 0:3].T

  # Start with point 1 translation.
  mobility_2[0:3, 0:3] = mobility_1[0:3, 0:3]

  # Add coupling block transpose cross r.
  mobility_2[0:3, 0:3] += tensor_cross_vector(mobility_1[0:3, 3:6 ], r)

  # Subtract r cross coupling block.
  mobility_2[0:3, 0:3] -= vector_cross_tensor(r, mobility_1[3:6, 0:3])

  # Subtract r cross D_r cross r
  mobility_2[0:3, 0:3] -= vector_cross_tensor(
     r, tensor_cross_vector(mobility_1[3:6, 3:6], r))

  return mobility_2


def tensor_cross_vector(T, v):
  ''' 
  Tensor cross vector from De La Torre paper.
  Assume T is 3x3 and v is length 3
  '''
  result = np.zeros([3,  3])
  
  for k in range(3):
    for l in range(3):
      result[k, l] = (T[k, (l+1) % 3]*v[(l - 1) % 3] - 
                      T[k, (l-1) % 3]*v[(l + 1) % 3])
  return result
   
   
def vector_cross_tensor(v, T):
  ''' 
  vector cross trensor from De La Torre paper.
  Assume T is 3x3 and v is length 3
  '''
  result = np.zeros([3,  3])
  for k in range(3):
    for l in range(3):
      result[k, l] = (T[(k-1) % 3, l]*v[(k + 1) % 3] - 
                      T[(k+1) % 3, l]*v[(k - 1) % 3])
  return result

@static_var('timers', {})   
def timer(name, print_one = False, print_all = False, clean_all = False):
  '''
  Timer to profile the code. It measures the time elapsed between successive
  calls and it prints the total time elapsed after sucesive calls.  
  '''
  if name not in timer.timers:
    timer.timers[name] = (0, time.time())
  elif timer.timers[name][1] is None:
    time_tuple = (timer.timers[name][0],  time.time())
    timer.timers[name] = time_tuple
  else:
    time_tuple = (timer.timers[name][0] + (time.time() - timer.timers[name][1]), None)
    timer.timers[name] = time_tuple
    if print_one is True:
      print(name, ' = ', timer.timers[name][0])

  if print_all is True:
    print('\n')
    col_width = max(len(key) for key in timer.timers)
    for key in sorted(timer.timers):
      print("".join(key.ljust(col_width)), ' = ', timer.timers[key][0])
      
  if clean_all:
    timer.timers = {}
  return


def gmres(A, b, x0=None, tol=1e-05, restart=None, maxiter=None, xtype=None, M=None, callback=None, restrt=None, PC_side='right'):
  '''
  Wrapper for scipy gmres to use right or left preconditioner.
  Solve the linear system A*x = b, using right or left preconditioning.
  Inputs and outputs as in scipy gmres plus PC_side ('right' or 'left').

  Right Preconditioner (default):
    First solve A*P^{-1} * y = b for y
    then solve P*x = y, for x.

  Left Preconditioner;
    Solve P^{-1}*A*x = P^{-1}*b


  Use Generalized Minimal Residual to solve A x = b.

  Parameters
  ----------
  A : {sparse matrix, dense matrix, LinearOperator}
      Matrix that defines the linear system.
  b : {array, matrix}
      Right hand side of the linear system. It can be a matrix.

  Returns
  -------
  x : {array, matrix}
      The solution of the linear system.
  info : int
      Provides convergence information:
        * 0  : success
        * >0 : convergence to tolerance not achieved, number of iterations
        * <0 : illegal input or breakdown

  Other parameters
  ----------------
  PC_side: {'right', 'left'}
      Use right or left Preconditioner. Right preconditioner (default) uses
      the real residual to determine convergence. Left preconditioner uses
      a preconditioned residual (M*r^n = M*(b - A*x^n)) to determine convergence.
  x0 : {array, matrix}
      Initial guess for the linear system (zero by default).
  tol : float
      Tolerance. The solver finishes when the relative or the absolute residual  
      norm are below this tolerance.
  restart : int, optional
      Number of iterations between restarts. 
      Default is 20.
  maxiter : int, optional
      Maximum number of iterations.  
  xtype : {'f','d','F','D'}
      This parameter is DEPRECATED --- avoid using it.
      The type of the result.  If None, then it will be determined from
      A.dtype.char and b.  If A does not have a typecode method then it
      will compute A.matvec(x0) to get a typecode.   To save the extra
      computation when A does not have a typecode attribute use xtype=0
      for the same type as b or use xtype='f','d','F',or 'D'.
      This parameter has been superseded by LinearOperator.
  M : {sparse matrix, dense matrix, LinearOperator}
      Inverse of the preconditioner of A. By default M is None.
  callback : function
      User-supplied function to call after each iteration.  It is called
      as callback(rk), where rk is the current residual vector.
  restrt : int, optional
      DEPRECATED - use `restart` instead.

  See Also
  --------
  LinearOperator

  Notes
  -----
  A preconditioner, P, is chosen such that P is close to A but easy to solve
  for. The preconditioner parameter required by this routine is
  ``M = P^-1``. The inverse should preferably not be calculated
  explicitly.  Rather, use the following template to produce M::

  # Construct a linear operator that computes P^-1 * x.
  import scipy.sparse.linalg as spla
  M_x = lambda x: spla.spsolve(P, x)
  M = spla.LinearOperator((n, n), M_x)
  '''

  # If left preconditioner (or no Preconditioner) just call scipy gmres
  if PC_side == 'left' or M is None:
    return scspla.gmres(A, b, M=M, x0=x0, tol=tol, atol=0, maxiter=maxiter, restart=restart, callback=callback)    

  # Create LinearOperator for A and P^{-1}
  A_LO = scspla.aslinearoperator(A)
  M_LO = scspla.aslinearoperator(M)

  # Define new LinearOperators P^{-1} * A and A * P^{-1}
  def PinvA(x,A,M):
    return M.matvec(A.matvec(x))

  def APinv(x,A,M):
    return A.matvec(M.matvec(x))

  # Select new linear operator
  if PC_side == 'left_res':
    A_new = PinvA
  elif PC_side == 'right':
    A_new = APinv

  # Set new linear operator
  A_partial = partial(A_new, A=A_LO, M=M_LO)
  A_partial_LO = scspla.LinearOperator((b.size, b.size), matvec = A_partial, dtype='float64') 
    
  # Modify RHS
  if PC_side == 'left_res':
    # b_new = P^{-1} * b
    b = M.matvec(b)
  
  # Solve system A_new * x = b
  (x, info) = scspla.gmres(A_partial_LO, b, x0=None, tol=tol, atol=0, maxiter=maxiter, restart=restart, callback=callback) 

  # Modify solution
  if PC_side == 'right':
    # Solve system P*x = y
    x = M_LO.matvec(x)
  
  # Return solution and info
  return x, info


def get_vectors_frame_body(vector, body=None, translate=True, rotate=True, transpose=False):
  '''
  Get vector in the frame of reference of the body.
  If body == None, use the lab frame of reference, i.e. do not translate or rotate anything.

  Inputs:
  vector = input vector to transform.
  body = body to use as frame of reference.
  translate = if translate==False, do not translate vectors but they can be rotated.
  rotate = if rotated==False, do not rotate vectors but they can be translated.

  Outputs:
  vector_frame = vector in the body frame  of reference.
  '''
  if body is not None:
    # Rotate to body frame of reference
    if rotate:
      R0 = body.orientation.rotation_matrix()
      R0 = R0.T if transpose else R0
      vector_frame = np.einsum('ij,kj->ki', R0, vector.reshape((vector.size // 3, 3)))
      
    else:
      vector_frame = np.copy(vector.reshape((vector.size // 3, 3)))

    # Translate to body frame of reference
    if translate:
      vector_frame += body.location
      
  else:
    vector_frame = np.copy(vector)

  return vector_frame

