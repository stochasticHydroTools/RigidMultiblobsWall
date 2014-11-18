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
import tetrahedron as tdn
import numpy as np
from quaternion import Quaternion
from quaternion_integrator import QuaternionIntegrator

def calc_rotational_msd(integrator, dt, n_steps):
  for k in range(n_steps):
    integrator.position = [Quaternion([1, 0, 0, 0])]
    integrator.additive_em_time_step(dt)

  msd = np.array([np.zeros(3) for _ in range(3)])
  for pos in integrator.path:
    u_hat = np.zeros(3)
    rot_matrix = pos[0].rotation_matrix()
    for i in range(3):
      e = np.zeros(3)
      e[i] = 1.
      u_hat += 0.5*np.cross(e, np.inner(rot_matrix, e))
    msd += np.outer(u_hat, u_hat)
  msd = msd/float(len(integrator.path))/dt
  return msd

      
if __name__ == "__main__":
  # Set masses.
  tdn.M1 = 1.0
  tdn.M2 = 1.0
  tdn.M3 = 1.0
  initial_position = [Quaternion([1./np.sqrt(2), 0, 1./np.sqrt(2), 0])]
  # Create Fixman Integrator.
  integrator = QuaternionIntegrator(tdn.tetrahedron_mobility,
                                           initial_position, 
                                           tdn.gravity_torque_calculator)

  msd_calculated = calc_rotational_msd(integrator, 
                                       float(sys.argv[1]), 
                                       int(sys.argv[2]))
  
  print "Calculated MSD is ", msd_calculated
  msd_theory = 2.*integrator.kT*tdn.tetrahedron_mobility(initial_position)
  print "Theoretical MSD is ", msd_theory
  rel_error = np.linalg.norm(msd_calculated - msd_theory)/np.linalg.norm(msd_theory)
  print "Relative Error is ", rel_error
