''' 
Estimate the rotational MSD based on:

u_hat(dt) = \sum_i u_i(0) cross u_i(dt)
  
msd slope = <u_hat_i u_hat_j>/dt
  
This should go to 2kBT * Mobility as dt -> 0.
Evaluate mobility at point with no torque, and take several steps to
get a curve of MSD(t).
'''
import os
import sys
sys.path.append('..')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import tetrahedron as tdn
import numpy as np
import cPickle
from quaternion import Quaternion
from quaternion_integrator import QuaternionIntegrator


class MSDStatistics(object):
  '''
  Class to hold the means and std deviations of the time dependent
  MSD for multiple schemes and timesteps.  data is a dictionary of
  dictionaries, holding runs indexed by scheme and timestep in that 
  order.
  '''
  def __init__(self, schemes, dts):
    self.data = {}

  def add_run(self, scheme_name, dt, run_data):
    ''' 
    Add a run.  Create the entry if need be. 
    run is organized as a list of 3 arrays: [time, mean, std]
    In that order.
    '''
    if scheme_name not in self.data:
      self.data[scheme_name] = dict()

    self.data[scheme_name][dt] = run_data

def calculate_msd_from_fixed_initial_condition(initial_orientation,
                                               scheme,
                                               dt,
                                               end_time,
                                               n_runs):
  ''' 
  Calculate MSD by starting at an initial condition, and doing short runs
  to time = end_time.
  Average over these trajectories to get the curve of MSD[1, 1] v. time.
  '''
  integrator = QuaternionIntegrator(tdn.tetrahedron_mobility,
                                    initial_orientation, 
                                    tdn.gravity_torque_calculator)
  n_steps = int(end_time/dt)
  trajectories = []
  for run in range(n_runs):
    integrator.orientation = initial_orientation
    trajectories.append([])
    #HACK: For now we just take the [1, 1] entry of the rotational MSD matrix.
    trajectories[run].append(
      calc_rotational_msd(initial_orientation[0],
                          integrator.orientation[0])[1, 1])
    for step in range(n_steps):

      if scheme == 'FIXMAN':
        integrator.fixman_time_step(dt)
      elif scheme == 'RFD':
        integrator.rfd_time_step(dt)
      elif scheme == 'EM':
        integrator.additive_em_time_step(dt)
      trajectories[run].append(
        calc_rotational_msd(initial_orientation[0],
                            integrator.orientation[0])[1, 1])
  # Average results to get time, mean, and std of rotational MSD.
  results = [[], [], []]
  for step in range(n_steps):
    time = dt*step
    mean_msd = np.mean([trajectories[run][step] for run in range(n_runs)])
    std_msd = np.std([trajectories[run][step] for run in range(n_runs)])
    results[0].append(time)
    results[1].append(mean_msd)
    results[2].append(std_msd/np.sqrt(n_runs))

  return results


def calc_rotational_msd_from_long_run(initial_orientation,
                                      scheme,
                                      dt,
                                      end_time,
                                      n_steps):
  ''' 
  Do one long run, and along the way gather statsitics
  about the average rotational Mean Square Displacement 
  by calculating it from time lagged data. 
  args:
    initial_orientation: list of length 1 quaternion where 
                 the run starts.  This shouldn't effect restuls.
    scheme: FIXMAN, RFD, or EM, scheme for the integrator to use.
    dt:  float, timestep used by the integrator.
    end_time: float, how much time to track the evolution of the MSD.
    n_steps:  How many total steps to take.
  '''
  integrator = QuaternionIntegrator(tdn.tetrahedron_mobility,
                                    initial_orientation, 
                                    tdn.gravity_torque_calculator)
  trajectory_length = int(end_time/dt)
  lagged_trajectory = []
  average_rotational_msd = np.zeros(trajectory_length)
  for step in range(n_steps):
    if scheme == 'FIXMAN':
      integrator.fixman_time_step(dt)
    elif scheme == 'RFD':
      integrator.rfd_time_step(dt)
    elif scheme == 'EM':
      integrator.additive_em_time_step(dt)


    lagged_trajectory.append(integrator.orientation[0])

    if len(lagged_trajectory) > trajectory_length:
      lagged_trajectory = lagged_trajectory[1:]
      for k in range(trajectory_length):
        # For now just choose the 1, 1 entry
        average_rotational_msd[k] += calc_rotational_msd(
          lagged_trajectory[0],
          lagged_trajectory[k])[1, 1]
    
  average_rotational_msd = average_rotational_msd/(n_steps - trajectory_length)
  # Average results to get time, mean, and std of rotational MSD.
  # For now, std = 0.  Will figure out a good way to calculate this later.
  results = [[], [], []]
  results[0] = np.arange(0, trajectory_length)*dt
  results[1] = average_rotational_msd
  results[2] = np.zeros(trajectory_length)
      
  return results

  
def calc_rotational_msd(initial_orientation, orientation):
  ''' 
  Calculate the rotational MSD from an initial configuration to
  a final orientation.  Orientations are given as single quaternion objects.
  '''  
  u_hat = np.zeros(3)
  rot_matrix = orientation.rotation_matrix()
  original_rot_matrix = initial_orientation.rotation_matrix()
  for i in range(3):
    e = np.zeros(3)
    e[i] = 1.
    u_hat += 0.5*np.cross(np.inner(original_rot_matrix, e),
                          np.inner(rot_matrix, e))
  return np.outer(u_hat, u_hat)

  
def plot_msd_convergence(dts, msd_list, names):
  ''' 
  Log-log plot of error in MSD v. dt.  This is for single
  step MSD compared to theoretical MSD slope (mobility).
  '''
  fig = pyplot.figure()
  ax = fig.add_subplot(1, 1, 1)
  for k in range(len(msd_list)):
    pyplot.plot(dts, msd_list[k], label=names[k])

  first_order = msd_list[0][0]*((np.array(dts)))/(dts[0])
  pyplot.plot(dts, first_order, 'k--', label='1st Order')
  pyplot.ylabel('Error')
  pyplot.xlabel('dt')
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.title('Error in Rotational MSD')
  ax.set_yscale('log')
  ax.set_xscale('log')
  pyplot.savefig('./plots/RotationalMSD.pdf')


if __name__ == "__main__":
  # Set masses and initial position.
  tdn.M1 = 0.1
  tdn.M2 = 0.1
  tdn.M3 = 0.1
  initial_orientation = [Quaternion([1., 0., 0., 0.])]
#  initial_position = [Quaternion([1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.), 0.])]
  schemes = ['FIXMAN', 'RFD', 'EM']
  dts = [32., 16., 8.]
  end_time = 128.
  n_runs = 15000

  msd_statistics = MSDStatistics(schemes, dts)
  for scheme in schemes:
    for dt in dts:
      run_data = calculate_msd_from_fixed_initial_condition(initial_orientation,
                                                            scheme,
                                                            dt,
                                                            end_time,
                                                            n_runs)
      # run_data = calc_rotational_msd_from_long_run(initial_orientation,
      #                                                   scheme,
      #                                                   dt,
      #                                                   end_time,
      #                                                   n_runs)
      msd_statistics.add_run(scheme, dt, run_data)
      print 'finished timestep ', dt, 'for scheme ', scheme
      os.stdout.flush()
      

  # Make directory for data if it doesn't exist.
  if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
    os.mkdir(os.path.join(os.getcwd(), 'data'))

  # Optional name for data provided    
  if len(sys.argv) > 3:
    data_name = './data/rot-msd-dt-%s-N-%d-%s.pkl' % (dts, n_runs, sys.argv[3])
  else:
    data_name = './data/rot-msd-dt-%s-N-%d.pkl' % (dts, n_runs)

  with open(data_name, 'wb') as f:
    cPickle.dump(msd_statistics, f)

#  plot_time_dependent_msd(msd_statistics)
#  plot_msd_convergence(dts, [msd_fixman, msd_rfd, msd_em],
#                       ['Fixman', 'RFD', 'EM'])
  
  # print "Calculated MSD is ", msd_calculated
  # msd_theory = 2.*integrator.kT*tdn.tetrahedron_mobility(initial_position)
  # print "Theoretical MSD is ", msd_theory
  # rel_error = np.linalg.norm(msd_calculated - msd_theory)/np.linalg.norm(msd_theory)
  # print "Relative Error is ", rel_error

