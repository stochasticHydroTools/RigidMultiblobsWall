''' Script to load a trajectory and calculate the mean square displacement in theta.'''


import argparse
import cPickle
import cProfile
import logging
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import pstats
import StringIO
import sys
sys.path.append('..')
import time

import boomerang as bm
from config_local import DATA_DIR
from quaternion_integrator.quaternion import Quaternion
from utils import read_trajectory_from_txt
from utils import StreamToLogger
from utils import log_time_progress


def calculate_theta_displacement_in_plane(quaternion_1, quaternion_2):
  ''' 
  Calculate theta displacement between orientations of the
  boomerang.  This projects the bisector in the second orientation
  into the plane of the boomerang at the first orientation, and
  measures the angles between bisectors in this plane.
  '''
  # Fake location, we only care about orientation
  location = [0., 0., 0.]
  r_vectors_1 = bm.get_boomerang_r_vectors_15(location, quaternion_1)
  bisector_1 = (r_vectors_1[0] + r_vectors_1[14])
  bisector_1 /= np.linalg.norm(bisector_1)

  normal_vector = np.cross(r_vectors_1[0], r_vectors_1[14])
  normal_vector /= np.linalg.norm(normal_vector)
  
  r_vectors_2 = bm.get_boomerang_r_vectors_15(location, quaternion_2)
  bisector_2 = (r_vectors_2[0] + r_vectors_2[14])
  bisector_2 /= np.linalg.norm(bisector_2)
  
  projected_bisector_2 = bisector_2 - normal_vector*(np.dot(
      normal_vector, bisector_2))
  projected_bisector_2 /= np.linalg.norm(projected_bisector_2)
  print 'orthogonal? ', np.dot(projected_bisector_2, normal_vector)
  print 'projected_bisector 2 is ', projected_bisector_2
  print 'norm of projected_bisector 2 is ', np.linalg.norm(projected_bisector_2)
  print 'bisector 1 is ', bisector_1
  print 'dot product: ', np.dot(bisector_1, projected_bisector_2)
  

  d_theta = (np.arccos(np.dot(bisector_1, projected_bisector_2))*
             np.sign(np.cross(bisector_1, projected_bisector_2)))
    
  print "d_theta is ", d_theta
  return d_theta

def calculate_theta_displacement(quaternion_1, quaternion_2):
  ''' 
  Calculate theta displacement between orientations of the
  boomerang by computing the bisector projected onto the 
  x-y plane.
  '''
  # Fake location, we only care about orientation
  location = [0., 0., 0.]
  r_vectors_1 = bm.get_boomerang_r_vectors_15(location, quaternion_1)
  bisector_1 = (r_vectors_1[0] + r_vectors_1[14])[0:2]
  bisector_1 /= np.linalg.norm(bisector_1)

  r_vectors_2 = bm.get_boomerang_r_vectors_15(location, quaternion_2)
  bisector_2 = (r_vectors_2[0] + r_vectors_2[14])[0:2]
  bisector_2 /= np.linalg.norm(bisector_2)
  
  d_theta = (np.arccos(np.dot(bisector_1, bisector_2))*
             np.sign(np.cross(bisector_1, projected_bisector_2)))

  return d_theta


def calc_theta_msd_from_trajectory(orientations, dt, end,
                                   burn_in = 0, trajectory_length = 100):
  ''' Calculate rotational and translational (6x6) MSD matrix given a dictionary of
  trajectory data.  Return a numpy array of 6x6 MSD matrices, one for each time.
  params:
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
  n_steps = len(orientations)
  thetas = [0.]
  d_thetas = []
  for k in range(n_steps - 1):
    d_theta = calculate_theta_displacement(
      Quaternion(orientations[k]), 
      Quaternion(orientations[k-1]))
    d_thetas.append(d_theta)
    if abs(d_theta) > 1.0:
      print "d_theta is ", d_theta
    thetas.append(thetas[k] + d_theta)
  
  # Plot thetas as a sanity check
  plt.clf()
  plt.plot(dt*np.arange(n_steps), thetas)
  plt.savefig(os.path.join('.', 'figures', 'temp-thetapath.pdf'))
  plt.clf()
  hist = np.histogram(d_thetas, bins=50, range=[-0.3, 0.3])
  plt.plot((hist[1][:-1] + hist[1][1:])/2., hist[0])
  plt.savefig(os.path.join('.', 'figures', 'temp-dthetahist.pdf'))

  if trajectory_length*data_interval > n_steps:
    raise Exception('Trajectory length is longer than the total run. '
                    'Perform a longer run, or choose a shorter end time.')
  print_increment = int(n_steps/20)
  average_rotational_msd = np.zeros(trajectory_length)
  lagged_rotation_trajectory = []
  start_time = time.time()
  for k in range(n_steps):
    if k > trajectory_length:
      for l in range(trajectory_length):
        d_theta  = thetas[k] - thetas[k-l]
        average_rotational_msd[l] += (d_theta**2)
    if (k % print_increment) == 0 and k > 0:
      print 'At step %s of %s' % (k, n_steps)
      print 'For this run, time status is:'
      elapsed = time.time() - start_time
      log_time_progress(elapsed, k, n_steps)

  average_rotational_msd = (average_rotational_msd/
                            (n_steps/data_interval - trajectory_length - 
                             burn_in/data_interval))
  
  return average_rotational_msd


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Calculate rotation and '
                                   'translation MSD from a trajectory '
                                   'generated by boomerang.py. '
                                   'This assumes the data files are named '
                                   'similar to the following: \n '
                                   'boomerang-trajectory-dt-0.1-N-'
                                   '100000-scheme-RFD-g-2-example-name-#.txt\n'
                                   'where # ranges from 1 to n_runs. '
                                   'boomerang.py uses this '
                                   'convention.')
  parser.add_argument('-scheme', dest='scheme', type=str, default='RFD',
                      help='Scheme of data to analyze.  Options are '
                      'RFD, FIXMAN, or EM.  Defaults to RFD.')
  parser.add_argument('-free', dest='free', type=str, default='',
                      help='Is this boomerang free in 3 space, or confined '
                      'near a wall?')
  parser.add_argument('-dt', dest='dt', type=float,
                      help='Timestep of runs to analyze.')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps taken in trajectory '
                      'data to be analyzed.')
  parser.add_argument('-gfactor', dest='gfactor', type=float,
                      help='Factor of earths gravity that simulation was '
                      'performed in.  For example 2 analyzes trajectories '
                      'from simulations that  are performed in double '
                      'earth gravity.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      help='Data name of trajectory runs to be analyzed.')
  parser.add_argument('-n_runs', dest='n_runs', type=int,
                      help='Number of trajectory runs to be analyzed.')
  parser.add_argument('-end', dest='end', type=float,
                      help='How far to analyze MSD (how large of a time window '
                      'to use).  This is in the same time units as dt.')
  parser.add_argument('--out-name', dest='out_name', type=str, default='',
                      help='Optional output name to add to the output Pkl '
                      'file for organization.  For example could denote '
                      'analysis using cross point v. vertex.')
  parser.add_argument('--profile', dest='profile', type=bool, default=False,
                      help='True or False: Do we profile this run or not. '
                      'Defaults to False. Put --profile 1 to profile.')
  

  # Some tests/investigation
  # theta_1 = Quaternion([1., 0., 0., 0.])
  # theta_2 = Quaternion([1./np.sqrt(2.), 1./np.sqrt(2.), 0., 0.])
  # d_theta = calculate_theta_displacement(theta_1, theta_2)
  # print "d_theta is ", d_theta

  # for k in range(100):
  #   theta_1 = np.random.normal(0., 1., 4)
  #   theta_1 = Quaternion(theta_1/np.linalg.norm(theta_1))
  #   theta_2 = np.random.normal(0., 1., 4)
  #   theta_2 = Quaternion(theta_2/np.linalg.norm(theta_2))
  #   d_theta = calculate_theta_displacement(theta_1, theta_2)
  #   print "d_theta is ", d_theta    
  args=parser.parse_args()
  if args.profile:
    pr = cProfile.Profile()
    pr.enable()


  # List files here to process.  They must have the same timestep, etc..
  scheme = args.scheme
  dt = args.dt
  end = args.end
  N = args.n_steps
  data_name = args.data_name
  trajectory_length = 40

  # Set up logging
  log_filename = 'boomerang-theta-msd-calculation-dt-%f-N-%d-g-%s-%s' % (
    dt, N, args.gfactor, args.data_name)
  if args.free:
    log_filename = 'free-' + log_filename
  if args.out_name:
    log_filename = log_filename + ('-%s' % args.out_name)
  log_filename = './logs/' + log_filename + '.log'

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
  
  trajectory_file_names = []
  for k in range(1, args.n_runs+1):
    if data_name:
      if args.free:
        trajectory_file_names.append(
          'free-boomerang-trajectory-dt-%g-N-%s-scheme-%s-%s-%s.txt' % (
            dt, N, scheme, data_name, k))
      else:
        trajectory_file_names.append(
        'boomerang-trajectory-dt-%g-N-%s-scheme-%s-g-%s-%s-%s.txt' % (
            dt, N, scheme, args.gfactor, data_name, k))
    else:
      if args.free:
        trajectory_file_names.append(
          'free-boomerang-trajectory-dt-%g-N-%s-scheme-%s-%s.txt' % (
            dt, N, scheme, k))
      else:
        trajectory_file_names.append(
          'boomerang-trajectory-dt-%g-N-%s-scheme-%s-g-%s-%s.txt' % (
            dt, N, scheme, args.gfactor, k))

  msd_runs = []
  ctr = 0
  for name in trajectory_file_names:
    ctr += 1
    data_file_name = os.path.join(DATA_DIR, 'boomerang', name)
    # Check correct timestep.
    params, _ , orientations = read_trajectory_from_txt(data_file_name)
    if (abs(float(params['dt']) - dt) > 1e-7):
      raise Exception('Timestep of data does not match specified timestep.')
    if int(params['n_steps']) != N:
      raise Exception('Number of steps in data does not match specified '
                      'Number of steps.')

    theta_msd_data = calc_theta_msd_from_trajectory(orientations, dt, end, 
                                                    trajectory_length=trajectory_length)

    msd_runs.append(theta_msd_data)
    print 'Completed run %s of %s' % (ctr, len(trajectory_file_names))

  mean_msd = np.mean(np.array(msd_runs), axis=0)
  std_msd = np.std(np.array(msd_runs), axis=0)/np.sqrt(len(trajectory_file_names))
  data_interval = int(end/dt/trajectory_length) + 1
  time = np.arange(0, len(mean_msd))*dt*data_interval


  out_data = [time, mean_msd, std_msd]
  # Save MSD data with pickle.
  if args.out_name:
    if args.free:
      msd_data_file_name = os.path.join(
        '.', 'data',
        'free-boomerang-theta-msd-dt-%s-N-%s-end-%s-scheme-%s-runs-%s-%s-%s.pkl' %
        (dt, N, end, scheme, len(trajectory_file_names), data_name,
         args.out_name))      
    else:
      msd_data_file_name = os.path.join(
        '.', 'data',
        'boomerang-theta-msd-dt-%s-N-%s-end-%s-scheme-%s-g-%s-runs-%s-%s-%s.pkl' %
        (dt, N, end, scheme, args.gfactor, len(trajectory_file_names), data_name,
         args.out_name))
  else:
    if args.free:
      msd_data_file_name = os.path.join(
        '.', 'data',
        'free-boomerang-theta-msd-dt-%s-N-%s-end-%s-scheme-%s-runs-%s-%s.pkl' %
        (dt, N, end, scheme, len(trajectory_file_names), data_name))
    else:
      msd_data_file_name = os.path.join(
        '.', 'data',
        'boomerang-theta-msd-dt-%s-N-%s-end-%s-scheme-%s-g-%s-runs-%s-%s.pkl' %
        (dt, N, end, scheme, args.gfactor, len(trajectory_file_names), data_name))

  with open(msd_data_file_name, 'wb') as f:
    cPickle.dump(out_data, f)

  if args.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()  
