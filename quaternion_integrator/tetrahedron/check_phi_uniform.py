''' 
Script to check that phi, the angle of rotation around the vertical axis, 
is uniformly distributed as it should be for a Fixed tetrahedron.
'''

import sys
sys.path.append('../..')
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

import quaternion_integrator.tetrahedron.tetrahedron as tdn
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from quaternion_integrator.quaternion import Quaternion

def bin_phi(orientation, bin_width, phi_hist):
  ''' Bin the angle phi given an orientation. '''
  R = orientation.rotation_matrix()
  # Rotated vector just in the X plane for calculating phi.
  rotated_x = np.array([R[0, 0], R[1, 0]])/np.sqrt(R[0, 0]**2 + R[1, 0]**2)
  phi= (np.arcsin(rotated_x[1]) + 
        (np.pi - 2.*np.arcsin(rotated_x[1]))*(rotated_x[0] < 0.))
  idx = int(math.floor((phi + np.pi/2.)/bin_width))
  phi_hist[idx] += 1

def plot_phis(phi_list, names, buckets):
  ''' 
  Plot phi distributions from a list of histograms. Each entry
  in the phi_list corresponds to one scheme.  The order of the entries
  should correspond to the order of names, a list of strings to label
  each scheme in the plot.
  '''
  # For now assume uniform buckets
  bin_width = buckets[1] - buckets[0]
  for k in range(len(phi_list)):
    pyplot.plot(buckets, phi_list[k]/float(sum(phi_list[k]))/bin_width,
                label = names[k])

  pyplot.ylim(0., 0.2)
  pyplot.legend(loc='best', prop={'size': 9})
  pyplot.savefig('./figures/PhiDistribution.pdf')

  
if __name__ == "__main__":
  # Make sure figures folder exists
  if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
    os.mkdir(os.path.join(os.getcwd(), 'figures'))

  # Script to run the various integrators on the quaternion.
  initial_orientation = [Quaternion([1., 0., 0., 0.])]
  fixman_integrator = QuaternionIntegrator(tdn.tetrahedron_mobility,
                                           initial_orientation, 
                                           tdn.gravity_torque_calculator)
  rfd_integrator = QuaternionIntegrator(tdn.tetrahedron_mobility, 
                                        initial_orientation, 
                                        tdn.gravity_torque_calculator)
  em_integrator = QuaternionIntegrator(tdn.tetrahedron_mobility, 
                                       initial_orientation, 
                                       tdn.gravity_torque_calculator)
  # Get command line parameters
  dt = float(sys.argv[1])
  n_steps = int(sys.argv[2])
  print_increment = max(int(n_steps/10.), 1)

  n_buckets = 20
  fixman_phi = np.zeros(n_buckets)
  rfd_phi = np.zeros(n_buckets)
  em_phi =  np.zeros(n_buckets)
  bin_width = 2*np.pi/n_buckets

  for k in range(n_steps):
    # Fixman step and bin result.
    fixman_integrator.fixman_time_step(dt)
    bin_phi(fixman_integrator.orientation[0], 
            bin_width, 
            fixman_phi)
    # RFD step and bin result.
    rfd_integrator.rfd_time_step(dt)
    bin_phi(rfd_integrator.orientation[0],
            bin_width, 
            rfd_phi)    
    # EM step and bin result.
    em_integrator.additive_em_time_step(dt)
    bin_phi(em_integrator.orientation[0],
            bin_width, 
            em_phi)

    if k % print_increment == 0:
      print "At step:", k

  names = ['Fixman', 'RFD', 'EM']
  bucket_boundaries = np.linspace(-0.5*np.pi, 1.5*np.pi, n_buckets + 1)
  buckets  = (bucket_boundaries[:-1] + bucket_boundaries[1:])/2.
  plot_phis([fixman_phi, rfd_phi, em_phi], names, buckets)
