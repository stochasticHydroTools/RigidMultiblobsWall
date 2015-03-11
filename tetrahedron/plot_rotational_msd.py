''' 
Plot rotational msd data from a pickle file. This is for final plots, 
and assumes that there could be multiple runs at each dt and scheme with 
the same number of steps.  It expects all data to be plotted to have the same
data-name with increasing numbers, e.g. *-run-1.pkl, *-run-2.pkl

run:
  python plot_rotational_msd.py -h
for usage.
'''
import argparse
import cPickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import os
import sys
sys.path.append('..')

from quaternion_integrator.quaternion import Quaternion
from translational_diffusion_coefficient import calculate_average_mu_parallel_and_perpendicular
import tetrahedron_free as tf
from tetrahedron_rotational_msd import calc_rotational_msd
from utils import MSDStatistics
from utils import plot_time_dependent_msd


def calculate_zz_and_rot_msd_at_equilibrium(n_steps):
  ''' 
  Calculate the zz (and rot) at equilibrium by generating pairs of samples
  and calculating their MSD.  We use this to compare to
  the ZZ and rotational MSD from the runs.
  '''
  zz_msd = 0.0
  rot_msd = 0.
  for k in range(n_steps):
    sample = tf.generate_free_equilibrium_sample()
    sample_2 = tf.generate_free_equilibrium_sample()
    center_1 = tf.get_free_center_of_mass(sample[0], sample[1])
    center_2 = tf.get_free_center_of_mass(sample_2[0], sample_2[1])
    zz_msd += (center_1[2] - center_2[2])**2.
    rot_matrix_1 = sample[1].rotation_matrix()
    rot_matrix_2 = sample_2[1].rotation_matrix()
    rot_msd += calc_rotational_msd(rot_matrix_1, rot_matrix_2)[0, 0]
  zz_msd /= n_steps
  rot_msd /= n_steps

  return [zz_msd, rot_msd]


def calculate_rot_msd_at_equilibrium(n_steps):
  ''' 
  Calculate the rotational (3-3 component) MSD at equilibrium by
  generating pairs of samples and calculating their rotational MSD.
  We use this to compare to the rotational MSD from the runs.
  '''
  rot_msd = 0.0
  for k in range(n_steps):
    sample = tf.generate_free_equilibrium_sample()
    sample_2 = tf.generate_free_equilibrium_sample()
    
#    center_1 = tf.get_free_center_of_mass(sample[0], sample[1])
#    center_2 = tf.get_free_center_of_mass(sample_2[0], sample_2[1])
    rot_msd += (center_1[2] - center_2[2])**2.
  rot_msd /= n_steps

  return zz_msd


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Plot results of Rotational MSD '
                                   'Simulations from pkl files created by '
                                   'tetrahedron_rotational_msd.py.')
  parser.add_argument('-dts', dest='dts', type=float, nargs = '+',
                      help='Timesteps to plot')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps taken.')
  parser.add_argument('-files', dest='n_files', type=int, default = None,
                      help='Number of data files at each step to combine. '
                      'This assumes that the data files are named *-1.pkl, '
                      '*-2.pkl, etc.')
  parser.add_argument('-schemes', dest='schemes', type=str, nargs='+',
                      help='Schemes to plot')
  parser.add_argument('-initial', dest='initial', type=bool, default=False,
                      help='If true, plot runs that start at one fixed initial '
                      'condition.  If False, plot runs that give equilibrium '
                      'MSD.')
  parser.add_argument('-free', dest='has_location', type=bool,
                      default=True,
                      help='If true, plot runs where Tetrahedron is allowed '
                      'to move.  If False, plot runs where Tetrahedron '
                      'is fixed.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      help='Name of data runs to plot.  All runs must have '
                      'the same name specified when running '
                      'tetrahedron_rotational_msd.py to plot together. '
                      ' This is easy to change by just renaming the pkl file.')
  args = parser.parse_args()

  combined_msd_statistics = None
  # Open data file.
  data_path = os.path.join(os.getcwd(), 'data')
  for dt in args.dts:
    for scheme in args.schemes:
      if args.n_files:
        time = None
        average_msd = None
        std_msd = None
        for k in range(args.n_files):
          data_file = ('rot-msd-initial-%s-location-%s-scheme-%s'
                       '-dt-%s-N-%s-%s-%s.pkl' % (
                         args.initial, args.has_location, scheme, dt, args.n_steps,
                         args.data_name, k+1))
          data_name = os.path.join('data', data_file)
          with open(data_name, 'rb') as f:
            msd_statistics = cPickle.load(f)
            msd_statistics.print_params()
          if time is None:
            time = msd_statistics.data[scheme][dt][0]
            average_msd = msd_statistics.data[scheme][dt][1]
            std_msd = msd_statistics.data[scheme][dt][2]**2
          else:
            average_msd += msd_statistics.data[scheme][dt][1]
            std_msd += msd_statistics.data[scheme][dt][2]**2
        
        average_msd /= float(args.n_files)
        std_msd = np.sqrt(std_msd)/float(args.n_files)
        run_data = [time, average_msd, std_msd]
        if not combined_msd_statistics:
          combined_msd_statistics = MSDStatistics(msd_statistics.params)
        print "adding run for dt = %s, scheme = %s" % (dt, scheme)
        combined_msd_statistics.add_run(scheme, dt, run_data)
      else:
        data_file = ('rot-msd-initial-%s-location-%s-scheme-%s'
                     '-dt-%s-N-%s-%s.pkl' % (
                       args.initial, args.has_location, scheme, dt, args.n_steps,
                       args.data_name))
        data_name = os.path.join('data', data_file)
        with open(data_name, 'rb') as f:
          msd_statistics = cPickle.load(f)
          msd_statistics.print_params()
        if not combined_msd_statistics:
          combined_msd_statistics = msd_statistics
        else:
          combined_msd_statistics.add_run(scheme, dt, msd_statistics.data[scheme][dt])

  # HACK, add xx and yy to get translational data
  for scheme in combined_msd_statistics.data:
    for dt in combined_msd_statistics.data[scheme]:
      for k in range(len(combined_msd_statistics.data[scheme][dt][1])):
        combined_msd_statistics.data[scheme][dt][1][k][0][0] = (
          combined_msd_statistics.data[scheme][dt][1][k][0][0] +
          combined_msd_statistics.data[scheme][dt][1][k][1][1])
        combined_msd_statistics.data[scheme][dt][2][k][0][0] = np.sqrt(
          combined_msd_statistics.data[scheme][dt][2][k][0][0]**2 +
          combined_msd_statistics.data[scheme][dt][2][k][1][1]**2)

  if args.has_location:
    average_mob_and_friction = calculate_average_mu_parallel_and_perpendicular(40)
    [zz_msd, rot_msd] = calculate_zz_and_rot_msd_at_equilibrium(40)
    #HACK, overwrite to compare to initial condition run.
    print "Mobility at initial location is ", tf.free_tetrahedron_mobility([[0., 0., 3.5]],
                                                               [Quaternion([1., 0., 0., 0.])])
    average_mob_and_friction[0] = tf.free_tetrahedron_mobility([[0., 0., 3.5]],
                                                               [Quaternion([1., 0., 0., 0.])])[0, 0]
  # Decide which components go on which figures.
  figure_numbers = [1, 5, 1, 2, 3, 4]
  labels= [' Parallel MSD', ' YY-MSD', ' Perpendicular MSD', ' Rotational MSD', ' Rotational MSD', ' Rotational MSD']
  styles = ['o', '^', 's', 'o', '.', '.']
  translation_end = 25.0
  for l in range(6):
    ind = [l, l]
    plot_time_dependent_msd(combined_msd_statistics, ind, figure_numbers[l],
                            error_indices=[0, 2, 3], label=labels[l], symbol=styles[l],
                            num_err_bars=40)
    pyplot.figure(figure_numbers[l])
    if args.has_location:
      if l in [0]:
        pyplot.plot([0.0, translation_end], 
                    [0.0, translation_end*4.*tf.KT*average_mob_and_friction[0]], 'k-',
                    label=r'Average Mobility')
      elif l == 2:
        pyplot.plot([0.0, translation_end],
                    [zz_msd, zz_msd], 'b--', label='Asymptotic Perpendicular MSD')
#        fit_line = np.polyfit([combined_msd_statistics.data['RFD'][1.6][0][_] for _ in range(5)],
#                              [combined_msd_statistics.data['RFD'][1.6][1][_][2][2] for _ in range(5)],
#                              1)
#        print "fit line is ", fit_line
#        print "ratio for perp is ", fit_line[0]/(2*tf.KT*average_mob_and_friction[2])

        pyplot.plot([0.0, translation_end],
                    [0.0, translation_end*2.*tf.KT*average_mob_and_friction[2]],
                    'b:', label='Average Perpendicular Mobility')
        pyplot.xlim([0., translation_end])
        pyplot.ylim([0., translation_end*4.*tf.KT*average_mob_and_friction[0]])
    if l == 3:
      pyplot.plot([0.0, 500.],
                  [rot_msd, rot_msd], 'k--', label='Asymptotic Rotational MSD')
      pyplot.xlim([0., 500.])
    pyplot.title('MSD(t) for Tetrahedron')
    pyplot.legend(loc='best', prop={'size': 11})
    pyplot.savefig('./figures/TimeDependentRotationalMSD-Component-%s-%s.pdf' % 
                   (l, l))

  print "Mu parallel on average is ", average_mob_and_friction[0]*2.
  print "Mu perp on average is ", average_mob_and_friction[2]
  print "MSD Perpendicular asymptotic is ", zz_msd
  print "MSD Rotational asymptotic is ", rot_msd
