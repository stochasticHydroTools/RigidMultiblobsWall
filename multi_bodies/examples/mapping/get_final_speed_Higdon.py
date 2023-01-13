'''
Measure velocity of a swimmer with a single flagellum using Higdon method.
See The hydrodynamics of flagellar propulsion: helical waves, J. J. Higdon, JFM (94) 331 (1979).

IMPORTANT: the velocity is printed in the laboratory frame of reference, 
           to obtain the velocity in the body frame of reference the initial
           orientation of the swimmer should be quaternion=(1,0,0,0).           

How to use:
1. Set parameters at the top of main.
2. Run as python get_final_speed.py file_velocity.dat

Parameters:
1. name = name of the file with the bodies velocity
2. output_name = file name to save the average velocity
'''

import numpy as np
import sys


if __name__ == '__main__':
  # Set parameters
  name = 'data/run_flagellated.bodies_velocities.dat'
  output_name = 'data/run_flagellated.bodies_velocities.average.dat'

  # Read input
  x = np.loadtxt(name)
  U_b = x[0, 0:3]
  omega_b = x[0,3:6]
  U_f = x[1,0:3]
  omega_f = x[1,3:6]

  # Print instantaneous velocity
  print('Instantaneous velocities')
  print('U_body              = ', U_b)
  print('omega_body          = ', omega_b, '\n')
  print('U_flagellum         = ', U_f)
  print('omega_flagellum     = ', omega_f, '\n')
  omega_diff = omega_f - omega_b
  speed_rotation_diff = np.linalg.norm(omega_diff)
  print('omega_diff          = ', omega_diff)
  print('speed_rotation_diff = ', speed_rotation_diff, '\n\n')


  # Compute average velocity with Higdon's method
  speed_b = np.dot(U_b, omega_f) / np.linalg.norm(omega_f)
  speed_rotation_b = np.dot(omega_b, omega_f) / np.linalg.norm(omega_f)
  U_frame = omega_f * speed_b / np.linalg.norm(omega_f)
  omega_frame = omega_f * speed_rotation_b / np.linalg.norm(omega_f)
  np.savetxt(output_name, np.hstack([U_frame, omega_frame]).reshape((1,6)))

  np.set_printoptions(precision=15)
  print('Average velocities')
  print('speed_b             = ', abs(speed_b))
  print('speed_rotation_b    = ', abs(speed_rotation_b))
  print('U_body_frame        = ', U_frame)
  print('omega_body_frame    = ', omega_frame)
  print('U*omega / |U||omega = ', np.dot(U_frame, omega_frame) / np.linalg.norm(U_frame) / np.linalg.norm(omega_frame))
  print()
  
  

