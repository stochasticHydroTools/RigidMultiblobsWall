''' Script to calculate equilibrium MSD from a given trajectory (or trajectories) for
the free tetrahedron.'''

import csv
import numpy as np
import os

import tetrahedron_free as tf
from utils import MSDStatistics

if __name__ == '__main__':
  
  # List files here to process.  They must have the same timestep.
  trajectory_file_names = ['']
  scheme = 'FIXMAN'
  dt = 1.6
  end = 1500.


  ##########
  msd_runs = np.array([])
  for name in trajectory_file_names:
    data_file_name = os.path.join(tf.DATA_DIR, name)
    with open(data_file_name) as f:
      reader = csv.reader(f, 'rb')
      trajectory_data = dict(x for x in reader)
    # Check correct timestep.
    if trajectory_data['dt'] != dt:
      raise Exception('Timestep of data does not match specified timestep.')

    # Calculate MSD data (just an array of MSD at each time.)
    msd_data = calc_msd_data_from_trajectory(trajectory_data)
    # append to calculate Mean and Std.
    msd_runs.append(msd)

  mean_msd = np.mean(msd_runs, axis=0)
  std_msd = np.std(msd_runs, axis=0)/np.sqrt(len(trajectory_file_names))
  time = np.arange(0, end, dt)

  params = {x: trajectory_data[x] for x in [
    'masses', 'KT', 'DEBYE_LENGTH', 'REPULSION_STRENGTH',
    'n_steps', 'dt', 'A', 'eta']}
  msd_statistics = MSDStatistics(params)
  msd_statistics.add_run(scheme, dt, [time, mean_msd, std_msd])

  # Save MSD data with pickle.

  
  
  
    
    
      
    
