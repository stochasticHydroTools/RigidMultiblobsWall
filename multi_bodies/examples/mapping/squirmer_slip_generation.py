'''
Set tangential slip for a spherical squirmer as

v_theta = B1 * sin(theta) + 0.5 * B2 * sin(2 * theta)

with theta the polar angle on the squirmer surface.

How to use:
1. Set parameters at the top of the main function.
2. Run as python squirmer_slip_generation.py

Outputs:
1. squirmer_N_N_B1_B1_B2_B2.slip: squirmer slip.
'''
import numpy as np
import math
import scipy as sp
import matplotlib as plt

if __name__ == '__main__':
  # Set parameters
  name_vertex = '../../Structures/shell_N_12_Rg_0_7921_Rh_1.vertex'
  output_prefix = './'
  B1 = 1
  B2 = 0

  # Read vertex file
  r_blobs = np.loadtxt(name_vertex, skiprows=1)

  # Extract coordinates
  x = r_blobs[:,0]
  y = r_blobs[:,1]
  z = r_blobs[:,2]

  # Compute spherical angles
  theta = np.arctan2(np.sqrt(x**2 + y**2), z)
  phi = np.arctan2(y, x)

  # Compute u_theta
  vs = B1 * np.sin(theta) + 0.5 * B2 * np.sin(2 * theta)

  # Compute velocity in cartesian coordinates
  v = np.zeros_like(r_blobs)
  v[:,0] = vs * np.cos(theta) * np.cos(phi)
  v[:,1] = vs * np.cos(theta) * np.sin(phi)
  v[:,2] = -vs * np.sin(theta) 

  # Save slip
  name = output_prefix + 'squirmer_N_' + str(x.shape[0]) + '_B1_' + str(B1) + '_B2_' + str(B2) + '.slip'
  with open(name, 'w') as f_handle:
    f_handle.write(str(v.shape[0]) + '\n')
    np.savetxt(f_handle, v, delimiter='\t')

  # Normal component of slip
  normal = np.zeros_like(r_blobs)
  normal = r_blobs / np.linalg.norm(r_blobs, axis=1)[:,None]

  un = np.einsum('bi,bi->b', normal, v)
  print('|normal_slip|_{\infty} = ', np.max(np.abs(un)))
