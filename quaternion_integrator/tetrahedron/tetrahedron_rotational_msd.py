''' 
Estimate the rotational MSD based on:

u_hat(dt) = \sum_i u_i(0) cross u_i(dt)
  
msd = <u_hat_i u_hat_j>/dt
  
This should go to 2kBT * Mobility as dt -> 0.
Evaluate mobility at point with no torque, in this case the reference configuration
when the particles have identical mass.
'''
import sys
sys.path.append('..')
from matplotlib import pyplot
import tetrahedron as tdn
import numpy as np
from quaternion import Quaternion
from quaternion_integrator import QuaternionIntegrator


def plot_msd_convergence(dts, msd_list, names):
  ''' Log-log plot of error in MSD v. dt '''
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
  tdn.M2 = 0.2
  tdn.M3 = 0.3
#  initial_position = [Quaternion([1., 0., 0., 0.])]
  initial_position = [Quaternion([1./np.sqrt(3.), 1./np.sqrt(3.), 1./np.sqrt(3.), 0.])]
  dts = [16., 8., 4., 2.]

  # Create Quaternion Integrator.
  integrator = QuaternionIntegrator(tdn.tetrahedron_mobility,
                                    initial_position, 
                                    tdn.gravity_torque_calculator)

  msd_fixman = []
  msd_rfd = []
  msd_em = []

  for dt in dts:
    msd_fixman.append(tdn.calc_rotational_msd(integrator, 
                                              "FIXMAN",
                                              dt, 
                                              int(sys.argv[1]),
                                              initial_position))

    msd_rfd.append(tdn.calc_rotational_msd(integrator, 
                                           "RFD",
                                           dt, 
                                           int(sys.argv[1]),
                                           initial_position))

    msd_em.append(tdn.calc_rotational_msd(integrator, 
                                          "EM",
                                          dt, 
                                          int(sys.argv[1]),
                                          initial_position))

  plot_msd_convergence(dts, [msd_fixman, msd_rfd, msd_em],
                       ['Fixman', 'RFD', 'EM'])
  
  # print "Calculated MSD is ", msd_calculated
  # msd_theory = 2.*integrator.kT*tdn.tetrahedron_mobility(initial_position)
  # print "Theoretical MSD is ", msd_theory
  # rel_error = np.linalg.norm(msd_calculated - msd_theory)/np.linalg.norm(msd_theory)
  # print "Relative Error is ", rel_error

