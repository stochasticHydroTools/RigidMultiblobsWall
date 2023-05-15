import argparse
import numpy as np
import scipy.linalg as scla
import scipy.sparse.linalg as spla
import subprocess
from functools import partial
import sys 
import time

# Find project functions
found_functions = False
path_to_append = ''  
while found_functions is False:
  try: 
    from read_input import read_vertex_file
    found_functions = True 
  except ImportError as exc:
    sys.stderr.write('Error: failed to import settings module ({})\n'.format(exc))
    path_to_append += '../'
    print('searching functions in path ', path_to_append)
    sys.path.append(path_to_append)
    if len(path_to_append) > 21: 
      print('\nProjected functions not found. Edit path in create_laplace_file.py')
      sys.exit()
if __name__ == '__main__':
  # Geometric parameters 
  W = 1
  La = 5
  Lb = 5
  a = W/2
  
  # Phoretic parameters
  alpha = 10 
  k = 0 
  surface_mobility = -10 
  
  # Number of blobs along the circumference 
  Ntheta = 4
  two_a = 2*a
  # Number of sections along the arm
  Na = int(round(La/two_a))
  La = Na*two_a
  print(La)
  # Number of sections along the body
  Nb = int(round(Lb/(two_a)))
  Lb = Nb*two_a
  # Total number of blobs
  Nblobs =(Nb + 2*Na)*Ntheta 
  print('Nblobs = ', Nblobs)

  # Builds the first arm with length La
  rblobs_section_xz = np.array([[-a,0,-a],[a,0,-a],[a,0,a],[-a,0,a]])
  normals_section_xz = np.copy(rblobs_section_xz)/(a*np.sqrt(2))
  rblobs_arm1 = np.zeros((Na*Ntheta,3))
  normals_arm1 = np.zeros((Na*Ntheta,3))
  weights_arm1 = np.ones((Na*Ntheta,1))*4*a**2
  for i in range(Na):
    rblobs_arm1[Ntheta*i:Ntheta*(i+1),:] = rblobs_section_xz
    rblobs_arm1[Ntheta*i:Ntheta*(i+1),1] = i*two_a
    normals_arm1[Ntheta*i:Ntheta*(i+1),:] = normals_section_xz
  # Modify normals and weights at corners
  normals_arm1[0:4,:] = np.array([[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]])/np.sqrt(3)
  weights_arm1[0:4,:] = 3*a**2
  normals_arm1[-1,:] = np.array([-1,1,1])/np.sqrt(3)
  weights_arm1[-1,:] = 3*a**2
  normals_arm1[-4,:] = np.array([-1,1,-1])/np.sqrt(3)
  weights_arm1[-4,:] = 3*a**2
  normals_arm1[-3,:] = np.array([0,1,-1])/np.sqrt(2)
  normals_arm1[-2,:] = np.array([0,1,1])/np.sqrt(2)
  normals_arm1[-6,:] = np.array([1,-1,1])/np.sqrt(3)
  weights_arm1[-6,:] = 5*a**2
  normals_arm1[-7,:] = np.array([1,-1,-1])/np.sqrt(3)
  weights_arm1[-7,:] = 5*a**2

  # Builds the second arm with length La
  rblobs_arm2 = np.copy(rblobs_arm1)
  normals_arm2 = np.zeros((Na*Ntheta,3))
  weights_arm2 = np.ones((Na*Ntheta,1))*4*a**2
  for i in range(Na):
    normals_arm2[Ntheta*i:Ntheta*(i+1),:] = normals_section_xz
  rblobs_arm2[:,0] += W + Lb + two_a
  rblobs_arm2[:,1] += La - W - two_a
  # Modify normals and weights at corners
  normals_arm2[0,:] = np.array([0,-1,-1])/np.sqrt(2)
  normals_arm2[1,:] = np.array([1,-1,-1])/np.sqrt(3)
  weights_arm2[1,:] = 3*a**2
  normals_arm2[2,:] = np.array([1,-1,1])/np.sqrt(3)
  weights_arm2[2,:] = 3*a**2
  normals_arm2[3,:] = np.array([0,-1,1])/np.sqrt(2)
  normals_arm2[4,:] = np.array([-1,1,-1])/np.sqrt(3)
  weights_arm2[4,:] = 5*a**2
  normals_arm2[7,:] = np.array([-1,1,1])/np.sqrt(3)
  weights_arm2[7,:] = 5*a**2
  normals_arm2[-4:,:] = np.array([[-1,1,-1],[1,1,-1],[1,1,1],[-1,1,1]])/np.sqrt(3)
  weights_arm2[-4:,:] = 3*a**2

  # Builds the body with length Lb
  rblobs_section_yz = np.array([[0,-a,-a],[0,a,-a],[0,a,a],[0,-a,a]])
  normals_section_yz = np.copy(rblobs_section_yz)/(a*np.sqrt(2))
  rblobs_body = np.zeros((Nb*Ntheta,3))
  normals_body = np.zeros((Nb*Ntheta,3))
  weights_body = np.ones((Na*Ntheta,1))*4*a**2
  for i in range(Nb):
    rblobs_body[Ntheta*i:Ntheta*(i+1),:] = rblobs_section_yz
    rblobs_body[Ntheta*i:Ntheta*(i+1),0] = W/2 + (i+1)*two_a
    normals_body[Ntheta*i:Ntheta*(i+1),:] = normals_section_yz  
  rblobs_body[:,1] += La - W - a

  # Positions
  rblobs = np.concatenate((rblobs_arm1,rblobs_body,rblobs_arm2), axis = 0)
  rcom = np.mean(rblobs,axis=0)

  rblobs[:,0] -= rcom[0]
  rblobs[:,1] -= rcom[1]
  rblobs[:,2] -= rcom[2]

  # Normals
  normals = np.concatenate((normals_arm1,normals_body,normals_arm2), axis = 0)

  # Weights
  weights = np.concatenate((weights_arm1,weights_body,weights_arm2), axis = 0)
  
  # Save file
  str_La = str(format(La,'.2f')).replace('.','_')
  str_Lb = str(format(Lb,'.2f')).replace('.','_')
  str_W = str(format(W,'.2f')).replace('.','_')
  filename = '../../Structures/sprinkler_N_' + str(Nblobs) + '_La_' + str_La +  '_Lb_' + str_Lb + '_W_' + str_W
  print(filename)
  with open(filename + '.vertex','w') as f:
    f.write(str(int(Nblobs)) + ' ' +  str(a) + ' 0 ' + '\n')
    for sublist in rblobs:
      f.write(' '.join([str(item) for item in sublist]) + '\n' )


  #### Creates laplace file 
  # Reaction rates
  k_vec = np.ones((Nblobs, 1)) * k
  # Emission rates
  alpha_vec = np.ones((Nblobs,1)) * alpha
  # Surface mobility
  surface_mobility_vec = np.ones((Nblobs,1))*surface_mobility
  to_save = np.concatenate((normals, k_vec, alpha_vec, surface_mobility_vec, weights), axis=1)
  str_k = str(format(k,'.2f')).replace('.','_')
  str_alpha = str(format(alpha,'.2f')).replace('.','_')
  str_mob = str(format(surface_mobility,'.2f')).replace('.','_').replace('-','m')
  filename_Laplace = filename + '_k_' + str_k + '_alpha_' + str_alpha + '_surf_mob_' + str_mob
  print(filename_Laplace)
  np.savetxt(filename_Laplace  + '.Laplace', to_save, header='Columns: normals, reaction rate, emitting rate, surface mobility, weights')


  ## Save paraview file 
  to_save_paraview = np.concatenate((rblobs, normals, weights, k_vec, alpha_vec, surface_mobility_vec),axis=1)  
  np.savetxt(filename_Laplace + '.csv',to_save_paraview, delimiter=", ",header='Columns: x, y, z, normals, weights, reaction rate, emitting rate, surface mobility')
