'''
Use the quaternion integrator on a simple problem that should
result in a uniform distribution.  Test with uniform analyzer.
'''
import sys
import numpy as np
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from quaternion_integrator.quaternion import Quaternion
import uniform_analyzer as ua

def diagonal_mobility(position):
  ''' A simple diagonal mobility. '''
  # de-reference position to get quaternion.
  theta = position[0]
  return np.array([
      [1.0 + theta.s**2, 0., 0.],
      [0., 1. + theta.p[0]**2, 0.],
      [0., 0., 1. + theta.p[2]**2]
      ])

def zero_torque_calculator(position):
  ''' No torque, uniform distribution. '''
  return np.zeros(3)


if __name__ == '__main__':
  initial_position = [Quaternion([1., 0., 0., 0.])]
  simple_integrator = QuaternionIntegrator(diagonal_mobility, 
                                           initial_position, 
                                           zero_torque_calculator)
  # Generate uniform samples to compare against.
  uniform_samples = []
  for k in range(int(sys.argv[1])):
    simple_integrator.fixman_time_step(0.1)
    x = np.random.normal(0., 1., 4)
    x = x/np.linalg.norm(x)
    uniform_samples.append(x)

  rotation_analyzer = ua.UniformAnalyzer(simple_integrator.path, "Fixman")
  samples_analyzer = ua.UniformAnalyzer(uniform_samples, "Samples")

  ua.compare_distributions([rotation_analyzer, samples_analyzer])
