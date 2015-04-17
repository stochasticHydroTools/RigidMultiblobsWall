''' 
Populate files with samples of the boomerang from
the Gibbs Boltzmann distribution.  Can do one wall with gravity and
repulsion, or two walls with just gravity.
'''

import argparse
import numpy as np
import os
import sys
import time

import boomerang as bm
from config_local import DATA_DIR
from utils import log_time_progress
from utils import set_up_logger
from utils import write_trajectory_to_txt


if __name__ == '__main__':
  # Get command line arguments.
  parser = argparse.ArgumentParser(description='Use accept-reject to generate '
                                   'MC samples of boomerang distribution using '
                                   ' parameters currently in boomerang.py')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of samples to generate.')
  parser.add_argument('-gfactor', dest='gravity_factor', type=float, default=1.0,
                      help='Factor to increase gravity by.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      default='',
                      help='Optional name added to the end of the '
                      'sample file for identifying different parameters.')

  args=parser.parse_args()

  # Set up logging.
  log_filename = './logs/boomerang-equilibrium-samples-N-%d-g-%s-%s.log' % (
    args.n_steps, args.gravity_factor, args.data_name)
  progress_logger = set_up_logger(log_filename)
  # progress_logger = logging.getLogger('Progress Logger')
  # progress_logger.setLevel(logging.INFO)
  # # Add the log message handler to the logger
  # logging.basicConfig(filename=log_filename,
  #                     level=logging.INFO,
  #                     filemode='w')
  # sl = StreamToLogger(progress_logger, logging.INFO)
  # sys.stdout = sl
  # sl = StreamToLogger(progress_logger, logging.ERROR)
  # sys.stderr = sl


  original_mass = np.array(bm.M)
  bm.M = original_mass*args.gravity_factor
  # Store parameters to write to file, so we know which 
  # parameters this distribution corresponds to.
  params = {'M': bm.M, 'REPULSION_STRENGTH': bm.REPULSION_STRENGTH,
            'ETA': bm.ETA, 'A': bm.A, 'KT': bm.KT,
            'DEBYE_LENGTH': bm.DEBYE_LENGTH,
            'G': args.gravity_factor, 'N_BLOBS': len(bm.M)}

  trajectory = [[], []]
  print_increment = int(args.n_steps/20)
  start_time = time.time()
  for k in range(args.n_steps):
    sample = bm.generate_boomerang_equilibrium_sample()
    trajectory[0].append(sample[0])
    trajectory[1].append(sample[1].entries)
    if (k % print_increment == 0):
      elapsed = time.time() - start_time
      log_time_progress(elapsed, k+1, args.n_steps)

  file_name = os.path.join(DATA_DIR, 'boomerang', 
                           'boomerang-samples-g-%s-%s.txt' % 
                           (args.gravity_factor, args.data_name))
  write_trajectory_to_txt(file_name, trajectory, params)
  

    


  





