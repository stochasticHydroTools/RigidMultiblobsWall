''' Script to do timestep refinement on the tetrahedron problem. '''

import sys
sys.path.append('..')
sys.path.append('../..')
import numpy as np
import tetrahedron as tdn
from matplotlib import pyplot
from quaternion import Quaternion
from quaternion_integrator import QuaternionIntegrator

def plot_refinement(dts, fixman_errs, rfd_errs):
  pass

def calculate_error(integrator_path, equilibrium_samples):
  pass 
  

dts = [2.0, 1.0, 0.5]
n_steps = 40000
print_increment = int(n_steps/10.)

# Generate Equilibrium distribution
equilibrium_samples = []  
for k in range(n_steps):
  equilibrium_samples.append([tdn.generate_equilibrium_sample()])
  

fixman_errs = []
rfd_errs = []
for dt in dts:
  initial_position = [Quaternion([1., 0., 0., 0.])]
  fixman_integrator = QuaternionIntegrator(tdn.tetrahedron_mobility, 
                                           initial_position, 
                                           tdn.gravity_torque_calculator)

  rfd_integrator = QuaternionIntegrator(tdn.tetrahedron_mobility, 
                                        initial_position, 
                                        tdn.gravity_torque_calculator)

  em_integrator = QuaternionIntegrator(tdn.tetrahedron_mobility, 
                                       initial_position, 
                                       tdn.gravity_torque_calculator)

  for k in range(n_steps):
    fixman_integrator.fixman_time_step(dt)
    rfd_integrator.rfd_time_step(dt)
#    em_integrator.additive_em_time_step(dt)
    if k % print_increment == 0:
      print "At step:", k
  
  fixman_errs.append(calculate_err(fixman_integrator.path, equilibrium_samples))
  rfd_errs.append(calculate_err(rfd_integrator.path, equilibrium_samples))
  
plot_refinement(dts, fixman_errs, rfd_errs)

  

