'''  Overwrite masses and make a torque calculator for a non-uniform icosohedron.
Running this as a script will bin equilibrium heights and theta for plotting.'''

import argparse
import cPickle
import logging
import numpy as np
import math
import sys
import time

import icosohedron as ic
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from utils import static_var
from utils import StreamToLogger
from utils import log_time_progress


M = [0.00/12. for _ in range(12)]
M[11] += 0.1

def nonuniform_torque_calculator(location, orientation):
  ''' Calculate torque based on a nonuniform icosohedron. '''
  r_vectors = ic.get_icosohedron_r_vectors(location[0], orientation[0])
  forces = []
  for mass in M:
    forces += [0., 0., -1.*mass]
  R = ic.calc_icosohedron_rot_matrix(r_vectors, location[0])
  return np.dot(R.T, forces)


@static_var('max_index', 0)
@static_var('max_theta_index', 0)
def bin_height_and_theta(location, orientation, bin_width, height_histogram, 
                         theta_width, theta_hist):
  ''' Bin the height (of the geometric center) of the icosohedron, and 
  theta, the angle between the vector to the heavy blob (last r_vector) 
  and the negative z axis.'''
  # Bin Theta.
  r_vectors  = ic.get_icosohedron_r_vectors(location, orientation)
  heavy_blob_vector =  (location - r_vectors[-1])
  heavy_blob_vector /= np.linalg.norm(heavy_blob_vector)
  theta = np.arccos(heavy_blob_vector[2])
  theta_idx = int(theta/theta_width)
  if theta_idx < len(theta_hist):
    theta_hist[theta_idx] += 1
  else:
    if theta_idx > bin_height_and_theta.max_theta_index:
      bin_height_and_theta.max_theta_index = theta_idx
      print "New maximum Theta Index  %d is beyond histogram length " % theta_idx

  # Bin location.  Report overshoots.
  idx = int(math.floor((location[2])/bin_width))
  if idx < len(height_histogram):
    height_histogram[idx] += 1
  else:
    if idx > bin_height_and_theta.max_index:
      bin_height_and_theta.max_index = idx
      print "New maximum Index  %d is beyond histogram length " % idx
  

if __name__ == '__main__':
  # Get command line arguments.
  parser = argparse.ArgumentParser(description='Run Simulation of nonuniform '
                                   'icosohedron with Fixman and RFD '
                                   'schemes, and bin the resulting '
                                   'height and theta distribution.')
  parser.add_argument('-dt', dest='dt', type=float,
                      help='Timestep to use for runs.')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps to take for runs.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      default='',
                      help='Optional name added to the end of the '
                      'data file.  Useful for multiple runs '
                      '(--data_name=run-1).')
  parser.add_argument('--profile', dest='profile', type=bool, default=False,
                      help='True or False: Do we profile this run or not.')

  args=parser.parse_args()
  if args.profile:
    pr = cProfile.Profile()
    pr.enable()


  # Get command line parameters
  dt = args.dt
  n_steps = args.n_steps
  print_increment = max(int(n_steps/20.), 1)

  # Set up logging.
  log_filename = './logs/nonuniform-icosohedron-dt-%f-N-%d-%s.log' % (
    dt, n_steps, args.data_name)
  progress_logger = logging.getLogger('Progress Logger')
  progress_logger.setLevel(logging.INFO)
  # Add the log message handler to the logger
  logging.basicConfig(filename=log_filename,
                      level=logging.INFO,
                      filemode='w')
  sl = StreamToLogger(progress_logger, logging.INFO)
  sys.stdout = sl
  sl = StreamToLogger(progress_logger, logging.ERROR)
  sys.stderr = sl

  # Script to run the various integrators on the quaternion.
  initial_location = [[0., 0., 1.5]]
  initial_orientation = [Quaternion([1., 0., 0., 0.])]
  fixman_integrator = QuaternionIntegrator(ic.icosohedron_mobility,
                                           initial_orientation, 
                                           nonuniform_torque_calculator, 
                                           has_location=True,
                                           initial_location=initial_location,
                                           force_calculator=
                                           ic.icosohedron_force_calculator)
  fixman_integrator.kT = ic.KT
  fixman_integrator.check_function = ic.icosohedron_check_function
  rfd_integrator = QuaternionIntegrator(ic.icosohedron_mobility,
                                        initial_orientation, 
                                        nonuniform_torque_calculator,
                                        has_location=True,
                                        initial_location=initial_location,
                                        force_calculator=
                                        ic.icosohedron_force_calculator)
  rfd_integrator.kT = ic.KT
  rfd_integrator.check_function = ic.icosohedron_check_function
  
  # Set up histogram for heights.
  bin_width = 1./5.
  fixman_heights = np.zeros(int(12./bin_width))
  rfd_heights = np.zeros(int(12./bin_width))

  theta_bin_width = 1./10.
  fixman_thetas = np.zeros(int(2.*np.pi/bin_width))
  rfd_thetas = np.zeros(int(2.*np.pi/bin_width))
  
  start_time = time.time()
  progress_logger.info('Starting run...')
  for k in range(n_steps):
    # Fixman step and bin result.
    fixman_integrator.fixman_time_step(dt)
    bin_height_and_theta(fixman_integrator.location[0],
                         fixman_integrator.orientation[0],
                         bin_width, 
                         fixman_heights,
                         theta_bin_width,
                         fixman_thetas)
    # RFD step and bin result.
    rfd_integrator.rfd_time_step(dt)
    bin_height_and_theta(rfd_integrator.location[0],
                         rfd_integrator.orientation[0],
                         bin_width, 
                         rfd_heights,
                         theta_bin_width,
                         rfd_thetas)

    if k % print_increment == 0 and k > 0:
      elapsed_time = time.time() - start_time
      log_time_progress(elapsed_time, k, n_steps)
  progress_logger.info('Finished Runs.')
  # Gather data to save.
  heights = [fixman_heights/(n_steps*bin_width),
             rfd_heights/(n_steps*bin_width)]
  thetas = [fixman_thetas/(n_steps*theta_bin_width),
             rfd_thetas/(n_steps*theta_bin_width)]

  height_data = dict()
  # Save parameters just in case they're useful in the future.
  # TODO: Make sure you check all parameters when plotting to avoid
  # issues there.
  height_data['params'] = {'A': ic.A, 'ETA': ic.ETA, 'VERTEX_A': ic.VERTEX_A, 'M': M, 
                           'REPULSION_STRENGTH': ic.REPULSION_STRENGTH,
                           'DEBYE_LENGTH': ic.DEBYE_LENGTH, 'KT': ic.KT,}
  height_data['heights'] = heights
  height_data['thetas'] = thetas
  height_data['buckets'] = (bin_width*np.array(range(len(fixman_heights)))
                            + 0.5*bin_width)
  height_data['theta_buckets'] = (theta_bin_width*np.array(range(len(fixman_thetas)))
                            + 0.5*theta_bin_width)
  height_data['names'] = ['Fixman', 'RFD']

  # Optional name for data provided    
  if len(args.data_name) > 0:
    data_name = './data/nonuniform-icosohedron-dt-%g-N-%d-%s.pkl' % (dt, n_steps, args.data_name)
  else:
    data_name = './data/nonuniform-icosohedron-dt-%g-N-%d.pkl' % (dt, n_steps)

  with open(data_name, 'wb') as f:
    cPickle.dump(height_data, f)
  
  if args.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()  





