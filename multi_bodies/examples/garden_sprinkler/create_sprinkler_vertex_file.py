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
 
  slip = 0
  inert_corners = 0
  inert_edges = 0
  # Nw is the number of blobs along the width
  # this parameter controls the resolution 
  Nw = 2

  a = W/float((Nw-1))*0.5
  print(Nw)
  print('a = ', a)
  
  # Phoretic parameters
  alpha = 1
  k = 0
  surface_mobility = 1
  
  two_a = 2*a
  # Number of sections along the arm (excluding the ends)
  Na = int(round(La/two_a))
  La = Na*two_a
  print('La = ', La)
  # Number of sections along the body
  Nb = int(round(Lb/(two_a)))
  Lb = Nb*two_a
  print('Lb = ', Lb)

  ### Builds  each face of the arm 1
  ## faces with normal = +/-ey
  Npts_y = Nw*Nw
  x = np.linspace(0,W,Nw)
  xx, zz = np.meshgrid(x,x)

  # face with n = -ey
  if slip == 1:
    slip_my = np.zeros((Npts_y,3))
  rblobs_face_my = np.concatenate((xx.reshape((Npts_y,1)),np.zeros((Npts_y,1)),zz.reshape((Npts_y,1))), axis=1) 
  normals_my = np.zeros((Npts_y,3))
  normals_my[:,1] = -1
  weights_my = np.ones((Npts_y,1))*4*a**2
  alpha_my = np.ones((Npts_y,1))*alpha
  k_my = np.ones((Npts_y,1))*k
  M_my = np.ones((Npts_y,1))*surface_mobility
  # correct edges
  normals_my[0:Nw-1,:] = np.array([0,-1,-1])/np.sqrt(2)
  normals_my[-Nw:-1,:] = np.array([0,-1,1])/np.sqrt(2)
  if inert_edges == 1:
   alpha_my[0:Nw-1] = 0
   k_my[0:Nw-1] = 0
   M_my[0:Nw-1] = 0
   alpha_my[-Nw:-1] = 0
   k_my[-Nw:-1] = 0
   M_my[-Nw:-1] = 0
    
  for i in range(Nw-1):
    normals_my[Nw*(i+1)-1,:] = np.array([1,-1,0])/np.sqrt(2)
    normals_my[Nw*i,:] = np.array([-1,-1,0])/np.sqrt(2)
    if inert_edges == 1:
     alpha_my[Nw*(i+1)-1] = 0
     k_my[Nw*(i+1)-1] = 0
     M_my[Nw*(i+1)-1] = 0
     alpha_my[Nw*i] = 0
     k_my[Nw*i] = 0
     M_my[Nw*i] = 0

 
  # correct corners
  normals_my[0,:] = np.array([-1,-1,-1])/np.sqrt(3)
  weights_my[0]= 3*a**2
  normals_my[Nw-1,:] = np.array([1,-1,-1])/np.sqrt(3)
  weights_my[Nw-1] = 3*a**2
  normals_my[-Nw,:] = np.array([-1,-1,1])/np.sqrt(3)
  weights_my[-Nw] = 3*a**2
  normals_my[-1,:] = np.array([1,-1,1])/np.sqrt(3)
  weights_my[-1] = 3*a**2
  if inert_corners == 1:
   alpha_my[0] = 0
   alpha_my[Nw-1] = 0
   alpha_my[-Nw] = 0
   alpha_my[-1] = 0
   k_my[0] = 0
   k_my[Nw-1] = 0
   k_my[-Nw] = 0
   k_my[-1] = 0
   M_my[0] = 0
   M_my[Nw-1] = 0
   M_my[-Nw] = 0
   M_my[-1] = 0

  # face with n = ey
  if slip == 1:
    slip_py = np.zeros((Npts_y,3))
  rblobs_face_py =np.copy(rblobs_face_my) 
  rblobs_face_py[:,1] = La  
  normals_py =  np.copy(normals_my)
  normals_py[:,1] = -normals_py[:,1]
  weights_py =  np.copy(weights_my)
  alpha_py =  np.copy(alpha_my)
  k_py = np.copy(k_my)
  M_py = np.copy(M_my)
  # correct corners and edges
  for i in range(1,Nw-1):
    normals_py[Nw*(i+1)-1,:] = np.array([0,1,0])
    if inert_edges == 1:
     alpha_py[Nw*(i+1)-1] = alpha
     k_py[Nw*(i+1)-1] = k
     M_py[Nw*(i+1)-1] = surface_mobility
  weights_py[-1] = 4*a**2 
  normals_py[-1,:] = np.array([0,1,1])/np.sqrt(2)
  weights_py[Nw-1] = 4*a**2 
  normals_py[Nw-1,:] = np.array([0,1,-1])/np.sqrt(2)

  ## faces with normal = +/-ez
  Npts_z = (Nw-2)*(Na-1)
  x = np.linspace(two_a,W-two_a,Nw-2)
  y = np.linspace(two_a,La-two_a,Na-1)
  xx, yy = np.meshgrid(x,y)
  # face with n = -ez
  if slip == 1:
    slip_mz = np.zeros((Npts_z,3))
  rblobs_face_mz = np.concatenate((xx.reshape((Npts_z,1)),yy.reshape((Npts_z,1)),np.zeros((Npts_z,1))), axis=1) 
  normals_mz = np.zeros((Npts_z,3))
  normals_mz[:,2] = -1
  weights_mz = np.ones((Npts_z,1))*4*a**2
  alpha_mz = np.ones((Npts_z,1))*alpha
  k_mz = np.ones((Npts_z,1))*k
  M_mz = np.ones((Npts_z,1))*surface_mobility
  # face with n = +ez
  if slip == 1:
    slip_pz = np.zeros((Npts_z,3))
  rblobs_face_pz = np.copy(rblobs_face_mz) 
  rblobs_face_pz[:,2] = W  
  normals_pz = - np.copy(normals_mz)
  weights_pz = np.copy(weights_mz)
  alpha_pz = np.ones((Npts_z,1))*alpha
  k_pz = np.ones((Npts_z,1))*k
  M_pz = np.ones((Npts_z,1))*surface_mobility


  ## face with normal = -ex
  Npts_mx = Nw*(Na-1)
  y = np.linspace(two_a,La-two_a,Na-1)
  z = np.linspace(0,W,Nw)
  yy, zz = np.meshgrid(y,z)
  if slip == 1:
    slip_mx = np.zeros((Npts_mx,3))
    slip_mx[:,1] = 10
  rblobs_face_mx = np.concatenate((np.zeros((Npts_mx,1)),yy.reshape((Npts_mx,1)),zz.reshape((Npts_mx,1))), axis=1) 
  normals_mx = np.zeros((Npts_mx,3))
  normals_mx[:,0] = -1
  weights_mx = np.ones((Npts_mx,1))*4*a**2
  alpha_mx = np.ones((Npts_mx,1))*alpha
  k_mx = np.ones((Npts_mx,1))*k
  M_mx = np.ones((Npts_mx,1))*surface_mobility
  # correct edges
  for i in range(Na-1):
    normals_mx[i,:] = np.array([-1,0,-1])/np.sqrt(2)
    normals_mx[(Nw-1)*(Na-1)+i,:] = np.array([-1,0,1])/np.sqrt(2)
    if inert_edges == 1:
      alpha_mx[i] = 0
      k_mx[i] = 0
      M_mx[i] = 0
      alpha_mx[(Nw-1)*(Na-1)+i] = 0
      k_mx[(Nw-1)*(Na-1)+i] = 0
      M_mx[(Nw-1)*(Na-1)+i] = 0
  if slip == 1:
    un = np.einsum('bi,bi->b', normals_mx, slip_mx)
    print('|normal_slip|_{\infty} = ', np.max(np.abs(un)))

  ## face with normal = +ex
  Na_m_Nw = Na-Nw+1  
  Npts_px = Nw*Na_m_Nw
  y = np.linspace(two_a,La-W,Na_m_Nw)
  z = np.linspace(0,W,Nw)
  yy, zz = np.meshgrid(y,z)
  if slip == 1:
    slip_px = np.zeros((Npts_px,3))
  rblobs_face_px = np.concatenate((np.zeros((Npts_px,1)),yy.reshape((Npts_px,1)),zz.reshape((Npts_px,1))), axis=1)
  rblobs_face_px[:,0] = W 
  normals_px = np.zeros((Npts_px,3))
  normals_px[:,0] = 1
  weights_px = np.ones((Npts_px,1))*4*a**2
  alpha_px = np.ones((Npts_px,1))*alpha
  k_px = np.ones((Npts_px,1))*k
  M_px = np.ones((Npts_px,1))*surface_mobility
  # correct edges and corners
  for i in range(Na_m_Nw):
    normals_px[i,:] = np.array([1,0,-1])/np.sqrt(2)
    normals_px[(Nw-1)*(Na_m_Nw)+i,:] = np.array([1,0,1])/np.sqrt(2)
    if inert_edges == 1:
      alpha_px[i] = 0
      k_px[i] = 0
      M_px[i] = 0
      alpha_px[(Nw-1)*(Na_m_Nw)+i] = 0
      k_px[(Nw-1)*(Na_m_Nw)+i] = 0
      M_px[(Nw-1)*(Na_m_Nw)+i] = 0
      
  
  for j in range(Nw-1):
    normals_px[(Na_m_Nw)*(j+1)-1,:] = np.array([1,-1,0])/np.sqrt(2)
    if inert_edges == 1:
      alpha_px[(Na_m_Nw)*(j+1)-1] = 0
      k_px[(Na_m_Nw)*(j+1)-1] = 0
      M_px[(Na_m_Nw)*(j+1)-1] = 0

  normals_px[Na_m_Nw-1,:] = np.array([1,-1,-1])/np.sqrt(3)
  weights_px[Na_m_Nw-1] = 5*a**2
  normals_px[-1,:] = np.array([1,-1,1])/np.sqrt(3)
  weights_px[-1] = 5*a**2
  if inert_corners == 1:
    alpha_px[Na_m_Nw-1] = 0
    k_px[Na_m_Nw-1] = 0
    M_px[Na_m_Nw-1] = 0
    alpha_px[-1] = 0
    k_px[-1] = 0
    M_px[-1] = 0

  
  # Add the two missing edges
  Npts_emz = Nw-2
  xemz = np.ones((Npts_emz,1))*W
  yemz = np.linspace(La-W+two_a,La-two_a,Npts_emz).reshape((Npts_emz,1))
  print('yemz = ', yemz)
  zemz = np.zeros((Npts_emz,1))
  rblobs_edge_mz = np.concatenate((xemz,yemz,zemz), axis=1) 
  if slip == 1:
    slip_emz = np.zeros((Npts_emz,3))
  normals_emz = np.zeros((Npts_emz,3)) 
  normals_emz[:,2] = -1
  weights_emz = np.ones((Npts_emz,1))*4*a**2
  alpha_emz = np.ones((Npts_emz,1))*alpha
  k_emz = np.ones((Npts_emz,1))*k
  M_emz = np.ones((Npts_emz,1))*surface_mobility

  xepz = np.ones((Npts_emz,1))*W
  yepz = np.linspace(La-W+two_a,La-two_a,Npts_emz).reshape((Npts_emz,1))
  zepz = np.ones((Npts_emz,1))*W
  rblobs_edge_pz = np.concatenate((xepz,yepz,zepz), axis=1) 
  if slip == 1:
    slip_epz = np.zeros((Npts_emz,3))
  normals_epz = np.zeros((Npts_emz,3)) 
  normals_epz[:,2] = 1
  weights_epz = np.ones((Npts_emz,1))*4*a**2
  alpha_epz = np.ones((Npts_emz,1))*alpha
  k_epz = np.ones((Npts_emz,1))*k
  M_epz = np.ones((Npts_emz,1))*surface_mobility

  # Collect positions, normals and weights of arm1
  rblobs_arm1 = np.concatenate((rblobs_face_my,rblobs_face_py,rblobs_face_mz,rblobs_face_pz,rblobs_face_mx,rblobs_face_px,rblobs_edge_mz,rblobs_edge_pz),axis=0)   
  normals_arm1 = np.concatenate((normals_my,normals_py,normals_mz,normals_pz,normals_mx,normals_px,normals_emz,normals_epz),axis=0)   
  weights_arm1 = np.concatenate((weights_my,weights_py,weights_mz,weights_pz,weights_mx,weights_px,weights_emz,weights_epz),axis=0)   
  alpha_arm1 = np.concatenate((alpha_my,alpha_py,alpha_mz,alpha_pz,alpha_mx,alpha_px,alpha_emz,alpha_epz),axis=0)   
  k_arm1 = np.concatenate((k_my,k_py,k_mz,k_pz,k_mx,k_px,k_emz,k_epz),axis=0)   
  M_arm1 = np.concatenate((M_my,M_py,M_mz,M_pz,M_mx,M_px,M_emz,M_epz),axis=0)   
  if slip == 1:
    slip_arm1 = np.concatenate((slip_my,slip_py,slip_mz,slip_pz,slip_mx,slip_px,slip_emz,slip_epz),axis=0)

  La_check = np.amax(rblobs_arm1[:,1]) - np.min(rblobs_arm1[:,1]) 
  W_check = np.amax(rblobs_arm1[:,0]) - np.min(rblobs_arm1[:,0]) 
  print('La_check = ',  La_check)
  print('W_check = ',  W_check)

  ### Builds  each face of the body
  ## faces with normal = +/-ey
  Npts_y = Nw*(Nb+1)
  x = np.linspace(W+two_a,W+two_a+Lb,Nb+1)
  z = np.linspace(0,W,Nw)
  xx, zz = np.meshgrid(x,z)

  # face with n = -ey
  if slip == 1:
    slip_my = np.zeros((Npts_y,3))
  rblobs_face_my = np.concatenate((xx.reshape((Npts_y,1)),np.ones((Npts_y,1))*(La-W),zz.reshape((Npts_y,1))), axis=1) 
  normals_my = np.zeros((Npts_y,3))
  normals_my[:,1] = -1
  weights_my = np.ones((Npts_y,1))*4*a**2
  alpha_my = np.ones((Npts_y,1))*alpha
  k_my = np.ones((Npts_y,1))*k
  M_my = np.ones((Npts_y,1))*surface_mobility
  # correct edges
  for i in range(Nb+1):
    normals_my[i,:] = np.array([0,-1,-1])/np.sqrt(2)
    normals_my[(Nw-1)*(Nb+1)+i,:] = np.array([0,-1,1])/np.sqrt(2)
    if inert_edges == 1:
      alpha_my[i] = 0
      k_my[i] = 0
      M_my[i] = 0
      alpha_my[(Nw-1)*(Nb+1)+i] = 0
      k_my[(Nw-1)*(Nb+1)+i] = 0
      M_my[(Nw-1)*(Nb+1)+i] = 0

  # face with n = ey
  rblobs_face_py = np.copy(rblobs_face_my) 
  rblobs_face_py[:,1] = La  
  normals_py = np.copy(normals_my)
  normals_py[:,1] = -normals_py[:,1]
  weights_py = np.copy(weights_my)
  alpha_py = np.copy(alpha_my)
  k_py = np.copy(k_my)
  M_py = np.copy(M_my)
  if slip == 1:
    slip_py = np.zeros((Npts_y,3))

  ## faces with normal = +/-ez
  Npts_z = (Nw-2)*(Nb+1)
  y = np.linspace(La-W+two_a,La-two_a,Nw-2)
  x = np.linspace(W+two_a,W+two_a+Lb,Nb+1)
  xx, yy = np.meshgrid(x,y)
  # face with n = -ez
  if slip == 1:
    slip_mz = np.zeros((Npts_z,3))
  rblobs_face_mz = np.concatenate((xx.reshape((Npts_z,1)),yy.reshape((Npts_z,1)),np.zeros((Npts_z,1))), axis=1) 
  normals_mz = np.zeros((Npts_z,3))
  normals_mz[:,2] = -1
  weights_mz = np.ones((Npts_z,1))*4*a**2
  alpha_mz = np.ones((Npts_z,1))*alpha
  k_mz = np.ones((Npts_z,1))*k
  M_mz = np.ones((Npts_z,1))*surface_mobility
  # face with n = +ez
  if slip == 1:
    slip_pz = np.zeros((Npts_z,3))
  rblobs_face_pz = np.copy(rblobs_face_mz) 
  rblobs_face_pz[:,2] = W  
  normals_pz = - np.copy(normals_mz)
  weights_pz = np.copy(weights_mz)
  alpha_pz = np.copy(alpha_mz)
  k_pz = np.copy(k_mz)
  M_pz = np.copy(M_mz)

  # Collect positions, normals and weights of body
  rblobs_body = np.concatenate((rblobs_face_my,rblobs_face_py,rblobs_face_mz,rblobs_face_pz),axis=0)
  normals_body = np.concatenate((normals_my,normals_py,normals_mz,normals_pz),axis=0)    
  weights_body = np.concatenate((weights_my,weights_py,weights_mz,weights_pz),axis=0)
  alpha_body = np.concatenate((alpha_my,alpha_py,alpha_mz,alpha_pz),axis=0)
  k_body = np.concatenate((k_my,k_py,k_mz,k_pz),axis=0)
  M_body = np.concatenate((M_my,M_py,M_mz,M_pz),axis=0)
  if slip == 1: 
    slip_body = np.concatenate((slip_my,slip_py,slip_mz,slip_pz),axis=0)    
 
  Lb_check = np.amax(rblobs_body[:,0]) - np.min(rblobs_body[:,0]) 
  W_check = np.amax(rblobs_body[:,1]) - np.min(rblobs_body[:,1]) 
  print('Lb_check = ',  Lb_check)
  print('W_check = ',  W_check)

  ### Builds  each face of the arm 2
  rblobs_arm2 = np.copy(rblobs_arm1)
  rblobs_arm2[:,0] -= W/2
  rblobs_arm2[:,1] -= La/2
  rblobs_arm2[:,0] = - rblobs_arm2[:,0]
  rblobs_arm2[:,1] = - rblobs_arm2[:,1]
  rblobs_arm2[:,0] += W/2
  rblobs_arm2[:,1] += La/2
  rblobs_arm2[:,0] += W+Lb+2*two_a
  rblobs_arm2[:,1] += La-W
  normals_arm2 = np.copy(normals_arm1)
  normals_arm2[:,0] = -normals_arm2[:,0] 
  normals_arm2[:,1] = -normals_arm2[:,1] 
  weights_arm2 = np.copy(weights_arm1)
  alpha_arm2 = np.copy(alpha_arm1)
  k_arm2 = np.copy(k_arm1)
  M_arm2 = np.copy(M_arm1)
  if slip == 1:
    slip_arm2 = np.copy(slip_arm1)
    slip_arm2[:,0] = - slip_arm2[:,0] 
    slip_arm2[:,1] = - slip_arm2[:,1] 

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
  print('np.sum(weights) =  ', np.sum(weights))

  # Emission rate
  alpha_vec = np.concatenate((alpha_arm1,alpha_body,alpha_arm2), axis = 0)

  # reaction rate
  k_vec = np.concatenate((k_arm1,k_body,k_arm2), axis = 0)

  # surface mobility
  surface_mobility_vec = np.concatenate((M_arm1,M_body,M_arm2), axis = 0)

  # slip
  if slip == 1:
    slip_tot = np.concatenate((slip_arm1,slip_body,slip_arm2), axis = 0)

  Nblobs = weights.shape[0] 
  print('Nblobs = ',Nblobs) 
 
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
  to_save = np.concatenate((normals, k_vec, alpha_vec, surface_mobility_vec, weights), axis=1)
  str_k = str(format(k,'.2f')).replace('.','_')
  str_alpha = str(format(alpha,'.2f')).replace('.','_')
  str_mob = str(format(surface_mobility,'.2f')).replace('.','_').replace('-','m')
  filename_Laplace = filename + '_k_' + str_k + '_alpha_' + str_alpha + '_surf_mob_' + str_mob
  if inert_corners == 1:
    filename_Laplace += '_inert_corners'
  if inert_edges == 1:
    filename_Laplace += '_inert_edges'
  print(filename_Laplace)
  np.savetxt(filename_Laplace  + '.Laplace', to_save, header='Columns: normals, reaction rate, emitting rate, surface mobility, weights')

  ## save slip file
  if slip == 1:
    filename_slip = filename + '.slip'
    with open(filename_slip, 'w') as f_handle:
      f_handle.write(str(int(Nblobs)) + '\n')
      np.savetxt(f_handle, slip_tot, delimiter='\t')
  


  ## Save paraview file
  a_vec = np.ones((Nblobs,1))*a
  to_save_paraview = np.concatenate((rblobs, normals, weights, k_vec, alpha_vec, surface_mobility_vec, a_vec),axis=1)
  str_header = 'Columns: x, y, z, nx, ny, nz, weights, reaction rate, emitting rate, surface mobility, blob_radius' 
  if slip == 1:
    to_save_paraview = np.concatenate((to_save_paraview, slip_tot), axis=1)
    str_header += ', slipx, slipy, slipz'  
  np.savetxt(filename_Laplace + '.csv',to_save_paraview, delimiter=", ", header=str_header)
