'''
Set up the mobility, torque, and force functions for a free version of
the Boomerang
from:
"Chakrabarty et. al - Brownian Motion of Boomerang Colloidal
Particles"

We put this boomerang very far from the wall, and apply no forces or torques.

This file defines several functions needed to simulate
the boomerang, and contains several parameters for the run.

Running this script will generate a boomerang trajectory
which can be analyzed with other python scripts in this folder.
'''

import argparse
import cProfile
import numpy as np
import logging
import os
import pstats
import StringIO
import sys
sys.path.append('..')
import time


import boomerang as bm
from config_local import DATA_DIR
from fluids import mobility as mb
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from utils import log_time_progress
from utils import static_var
from utils import StreamToLogger
from utils import write_trajectory_to_txt

# Make sure figures folder exists
if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
  os.mkdir(os.path.join(os.getcwd(), 'figures'))
# Make sure logs folder exists
if not os.path.isdir(os.path.join(os.getcwd(), 'logs')):
  os.mkdir(os.path.join(os.getcwd(), 'logs'))


# Parameters.  Units are um, s, mg.
A = 0.265*np.sqrt(3./2.)   # Radius of blobs in um
ETA = 8.9e-4  # Water. Pa s = kg/(m s) = mg/(um s)

# 0.2 g/cm^3 = 0.0000000002 mg/um^3.  Volume is ~1.0238 um^3.  Include gravity in this.
# density of particle = 0.2 g/cm^3 = 0.0000000002 mg/um^3.  
# Volume is ~1.1781 um^3.  Include gravity in this.
TOTAL_MASS = 1.1781*0.0000000002*(9.8*1.e6)
M = [TOTAL_MASS/15. for _ in range(15)]
KT = 300.*1.3806488e-5  # T = 300K

# Made these up somewhat arbitrarily
REPULSION_STRENGTH = 7.5*KT
DEBYE_LENGTH = 0.5*A


def boomerang_free_force_calculator(location, orientation):
  ''' 
  Exert no force on the free boomerang.
  '''
  return np.array([0., 0., 0.])

def boomerang_free_torque_calculator(location, orientation):
  ''' 
  Exert no torque on the free boomerang.
  '''
  return np.array([0., 0., 0.])

if __name__ == '__main__':
  # Get command line arguments.
  parser = argparse.ArgumentParser(description='Run Simulation of Boomerang '
                                   'particle with Fixman, EM, and RFD '
                                   'schemes, and save trajectory.  Boomerang '
                                   'is affected by gravity and repulsed from '
                                   'the wall gently.')
  parser.add_argument('-dt', dest='dt', type=float,
                      help='Timestep to use for runs.')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps to take for runs.')
  parser.add_argument('-scheme', dest='scheme', type=str, default='RFD',
                      help='Numerical Scheme to use: RFD, FIXMAN, or EM.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      default='',
                      help='Optional name added to the end of the '
                      'data file.  Useful for multiple runs. '
                      'To analyze multiple runs and compute MSD, you must '
                      'specify this, and it must end with "-#" '
                      ' for # starting at 1 and increasing successively. e.g. '
                      'heavy-masses-1, heavy-masses-2, heavy-masses-3 etc.')
  parser.add_argument('--profile', dest='profile', type=bool, default=False,
                      help='True or False: Profile this run or not.')

  args=parser.parse_args()
  if args.profile:
    pr = cProfile.Profile()
    pr.enable()

  # Get command line parameters
  dt = args.dt
  n_steps = args.n_steps
  print_increment = max(int(n_steps/20.), 1)

  # Set up logging.
  log_filename = './logs/free-boomerang-dt-%f-N-%d-scheme-%s-%s.log' % (
    dt, n_steps, args.scheme, args.data_name)
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

  # Gather parameters to save
  params = {'A': bm.A, 'ETA': bm.ETA,
            'dt': dt, 'n_steps': n_steps,
            'scheme': args.scheme,
            'KT': bm.KT}

  print "Parameters for this run are: ", params

  # Script to run the various integrators on the quaternion.
  initial_location = [[0., 0., 900000000000000000.]]
  theta = np.random.normal(0., 1., 4)
  initial_orientation = [Quaternion(theta/np.linalg.norm(theta))]
  quaternion_integrator = QuaternionIntegrator(bm.boomerang_mobility,
                                               initial_orientation, 
                                               boomerang_free_torque_calculator,
                                               has_location=True,
                                               initial_location=initial_location,
                                               force_calculator=
                                               boomerang_free_force_calculator)
  quaternion_integrator.kT = bm.KT

  trajectory = [[], []]

  if len(args.data_name) > 0:
    def generate_trajectory_name(scheme):
      trajectory_dat_name = 'free-boomerang-trajectory-dt-%g-N-%d-scheme-%s-%s.txt' % (
        dt, n_steps, scheme, args.data_name)
      return trajectory_dat_name
  else:
    def generate_trajectory_name(scheme):
      trajectory_dat_name = 'free-boomerang-trajectory-dt-%g-N-%d-scheme-%s.txt' % (
        dt, n_steps, scheme)
      return trajectory_dat_name

  data_file = os.path.join(
    DATA_DIR, 'boomerang', generate_trajectory_name(args.scheme))
  write_trajectory_to_txt(data_file, trajectory, params)


  # First check that the directory exists.  If not, create it.
  dir_name = os.path.dirname(data_file)
  if not os.path.isdir(dir_name):
     os.mkdir(dir_name)

  # Write data to file, parameters first then trajectory.
  with open(data_file, 'w', 1) as f:
    f.write('Parameters:\n')
    for key, value in params.items():
      f.writelines(['%s: %s \n' % (key, value)])
    f.write('Trajectory data:\n')
    f.write('Location, Orientation:\n')

    start_time = time.time()
    for k in range(n_steps):
      # Fixman step and bin result.
      if args.scheme == 'FIXMAN':
        quaternion_integrator.fixman_time_step(dt)
      elif args.scheme == 'RFD':
        quaternion_integrator.rfd_time_step(dt)
      elif args.scheme == 'EM':
        # EM step and bin result.
        quaternion_integrator.additive_em_time_step(dt)
      else:
        raise Exception('scheme must be one of: RFD, FIXMAN, EM.')

#      trajectory[0].append(quaternion_integrator.location[0])
#      trajectory[1].append(quaternion_integrator.orientation[0].entries)
      location = quaternion_integrator.location[0]
      orientation = quaternion_integrator.orientation[0].entries
      f.write('%s, %s, %s, %s, %s, %s, %s \n' % (
        location[0], location[1], location[2], 
        orientation[0], orientation[1], orientation[2], orientation[3]))


      if k % print_increment == 0:
        elapsed_time = time.time() - start_time
        print 'At step %s out of %s' % (k, n_steps)
        log_time_progress(elapsed_time, k, n_steps)
      

  elapsed_time = time.time() - start_time
  if elapsed_time > 60:
    progress_logger.info('Finished timestepping. Total Time: %.2f minutes.' % 
                         (float(elapsed_time)/60.))
  else:
    progress_logger.info('Finished timestepping. Total Time: %.2f seconds.' % 
                         float(elapsed_time))

  progress_logger.info('Integrator Rejection rate: %s' % 
                       (float(quaternion_integrator.rejections)/
                        float(quaternion_integrator.rejections + n_steps)))

  if args.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

 
