'''
Set up the mobility, torque, and force functions for the Rod
from:
"Megan S. Davies Wykes et al. 2015"

This file defines several functions needed to simulate
the rod, and contains several parameters for the run.

Running this script will generate a rod trajectory
which can be analyzed with other python scripts in this folder.
'''

import argparse
import cProfile
import numpy as np
import logging
import os
import pstats
import StringIO
import sys
sys.path.append('..')
import time
import scipy.linalg as sla
import math as m

from config_local import DATA_DIR
from mobility import mobility as mb
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator_gmres import QuaternionIntegratorGMRES
from utils import log_time_progress
from utils import static_var
from utils import StreamToLogger
from utils import Tee
from utils import write_trajectory_to_txt

# Make sure figures folder exists
if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
  os.mkdir(os.path.join(os.getcwd(), 'figures'))
# Make sure logs folder exists
if not os.path.isdir(os.path.join(os.getcwd(), 'logs')):
  os.mkdir(os.path.join(os.getcwd(), 'logs'))


resolution = 3
print "resolution = ", resolution
# Parameters.  Units are um, s, mg.

# TO ADD AN IF STATEMENT TO CHOSE BETWEEN SHELLS AND CYLINDERS
#if resolution == 0:
  #A = 0.183228708092682 # To Match true velocities with Rh = 0.1623
  #Nblobs_per_rod = 14
#elif resolution == 1:
 ##A =  0.042116364993415 # To match true cylinder with a/s = 0.26515
 ##Nblobs_per_rod = 96
 #A =  0.0742 # To match true cylinder with a/s = 0.5, Rh=0.1623
 #Nblobs_per_rod =86
#elif resolution == 2:
 ##A = 0.021508896861336 # To match true cylinder with a/s = 0.25793
 ##Nblobs_per_rod = 374
 #A =  0.0402 # To match true cylinder with a/s = 0.5, Rh=0.1623
 #Nblobs_per_rod = 324
 
# BLOB RADII FOR SHELL WITH Rg = 0.225
if resolution == 1: 
  A = 0.1183 # 12 blobs
  Nblobs_per_rod = 12
elif resolution== 2:
  A= 0.061484985366573 # 42 blobs
  Nblobs_per_rod = 42
elif resolution== 3: 
  A = 0.031039252100254 # 162 blobs
  Nblobs_per_rod = 162
elif resolution == 4: 
  A = 0.015556856646539 # 642 blobs
  Nblobs_per_rod = 642
elif resolution == 5: 
  A = 0.007783085222670 # 642 blobs
  Nblobs_per_rod = 2562
  
DIAM_BLOB = 2.0*A  # Diameter of the blobs
if A == 0.07:
  DIAM_ROD_GEO = 0.28  # Geometric Diameter of the blobs
else:
  DIAM_ROD_GEO = 0.3246  # Geometric Diameter of the blobs
if resolution ==0:
 DIAM_ROD_EXCLU = DIAM_ROD_GEO
else: 
 DIAM_ROD_EXCLU = DIAM_ROD_GEO + DIAM_BLOB  # Excluded volume Diameter of the blobs

ETA = 1e-3  # Water. Pa s = kg/(m s) = mg/(um s)


# density of particle = 0.2 g/cm^3 = 0.0000000002 mg/um^3.  
# Volume is ~1.1781 um^3. 
TOTAL_MASS = 0*1.1781*0.0000000002*(9.8*1.e6)
M = [TOTAL_MASS/float(Nblobs_per_rod) for _ in range(Nblobs_per_rod)]
KT = 1.0*0.1*1.3806488e-5# 300.*1.3806488e-5  # T = 300K

# Made these up somewhat arbitrarily
REPULSION_STRENGTH_WALL = 0.0*20 * 300.*1.3806488e-5 #KT # 7.5*....
DEBYE_LENGTH_WALL = 0.5*A # 0.1*A

# Made these up somewhat arbitrarily
REPULSION_STRENGTH_BLOBS = 0.0*2. * 300.*1.3806488e-5 #KT
DEBYE_LENGTH_BLOBS = 0.005*A


def rod_mobility(r_vectors, rotation_matrix):
  ''' 
  Calculate the force and torque mobility for the
  rod.  Here location is the cross point.
  '''
  return force_and_torque_rod_mobility(r_vectors, rotation_matrix)

def force_and_torque_rod_mobility(r_vectors, rotation_matrix):
  '''
  Calculate the mobility: (torque, force) -> (angular velocity, velocity) at position 
  In this case, position has orientation and location data, each of length 1.
  The mobility is equal to the inverse of: 
    [ J^T M^-1 J,   J^T M^-1 R ]
    [ R^T M^-1 J,   R^T M^-1 R ]
  where R is 3N x 3 (18 x 3) Rx = r cross x and J is a 3N x 3 matrix with 
  each 3x3 block being the identity.
  r is the distance from the apex blob of the rod to
  each other blob (a length 3N vector).
  M (3N x 3N) is the finite size single wall mobility taken from the
  Swan and Brady paper:
   "Simulation of hydrodynamically interacting particles near a no-slip
    boundary."
  Here location is the dereferenced list with 3 entries.
  '''  
  # Blobs mobility
  if len(r_vectors) == 1: 
    r_vec_for_mob = r_vectors[0]
  else: 
    r_vec_for_mob = []
    for k in range(len(r_vectors)):
      r_vec_for_mob += r_vectors[k]

  mobility = mb.boosted_single_wall_fluid_mobility(r_vec_for_mob, ETA, A)

  # K matrix
  Nbody = len(r_vectors)
  Nblobs_per_body = len(r_vectors[0])
  J_tot = None
  for k in range(Nbody):
    J = np.concatenate([np.identity(3) for \
                     _ in range(Nblobs_per_body)])

    J_rot_combined = np.concatenate([J, rotation_matrix[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body,:]], axis=1)

    if J_tot is None:
        J_tot = J_rot_combined
    else:
        J_tot=np.concatenate([J_tot, J_rot_combined], axis=1)

  #print "rotation_matrix = ", 
  #print rotation_matrix
  #raw_input()
  mob_inv = np.linalg.inv(mobility)

  total_resistance = np.zeros((6*Nbody,6*Nbody))
  
  # ONLY BUILD UPPER TRIANGULAR PART OF R AND ASSIGN THE REST
  for k in range(Nbody):
    for j in range(k,Nbody):
  
        # VF BLOCK
        total_resistance[3*k:3*(k+1),3*j:3*(j+1)] =\
             np.dot(np.dot(J_tot[:,6*k:6*k+3].T,\
                    mob_inv[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body,3*j*Nblobs_per_body:3*(j+1)*Nblobs_per_body]),\
                    J_tot[:,6*j:6*j+3])
        if j>k:
	  total_resistance[3*j:3*(j+1),3*k:3*(k+1)] = \
                    total_resistance[3*k:3*(k+1),3*j:3*(j+1)].T

        # VT BLOCK
        total_resistance[3*k:3*(k+1),3*(Nbody+j):3*(Nbody+j+1)] =\
             np.dot(np.dot(J_tot[:,6*k:6*k+3].T,\
                    mob_inv[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body,3*j*Nblobs_per_body:3*(j+1)*Nblobs_per_body]),\
                    J_tot[:,6*j+3:6*(j+1)])
        if j>k:
            total_resistance[3*j:3*(j+1),3*(Nbody+k):3*(Nbody+k+1)] =\
                     np.dot(np.dot(J_tot[:,6*j:6*j+3].T,\
                    mob_inv[3*j*Nblobs_per_body:3*(j+1)*Nblobs_per_body,3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body]),\
                    J_tot[:,6*k+3:6*(k+1)])

        # WT BLOCK
        total_resistance[3*(Nbody+k):3*(Nbody+k+1),3*(Nbody+j):3*(Nbody+j+1)] =\
             np.dot(np.dot(J_tot[:,6*k+3:6*(k+1)].T,\
                    mob_inv[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body,3*j*Nblobs_per_body:3*(j+1)*Nblobs_per_body]),\
                    J_tot[:,6*j+3:6*(j+1)])
        if j>k:
	  total_resistance[3*(Nbody+j):3*(Nbody+j+1),3*(Nbody+k):3*(Nbody+k+1)] =\
                    total_resistance[3*(Nbody+k):3*(Nbody+k+1),3*(Nbody+j):3*(Nbody+j+1)].T

  # WF BLOCK IS JUST THE TRANSPOSE OF VT BLOCK
  total_resistance[3*Nbody:6*Nbody,0:3*Nbody] = total_resistance[0:3*Nbody,3*Nbody:6*Nbody].T
  
  
  #print "total_resistance - total_resistance.T = 0 ", \
         #np.allclose(total_resistance,total_resistance.T)
  #raw_input()

  # Mobility body
  total_mobility = np.linalg.pinv(total_resistance)
  #print "total_mobility = ", 
  #print total_mobility
  #raw_input()

  return (total_mobility, mob_inv)


def get_rod_initial_config(location, orientation):
  '''
  Depends on the resolution chosen
  ''' 
  
  folder_rods = 'Generated_rods/'
 
  if resolution == 0:
    initial_configuration = []
    # Using Bringley's formula for Nb = floor(Lrod/a_rod) + 1
    with open(folder_rods + 'Cylinder_l_geo_1.9295_radius_0.18323_Nblobs_perimeter_1_Nblobs_total_14_a_s_1.2345.vertex') as f:
      k = 0
      for l in f:
	k = k+1
	if k>1:
	  initial_configuration.append(np.array([float(x) for x in l.strip().split("  ")]))    
                              
  elif resolution == 1:
    initial_configuration = []
    # If D_eff = D = 0.3246 and a/s = 0.5
    #with open('Cylinder_l_geo_2.12_radius_0.1623_Nblobs_perimeter_6_Nblobs_total_86.vertex') as f:
    # If D_eff = D = 0.3246 and a/s = 0.2
    #with open('Cylinder_l_geo_2.12_radius_0.1623_Nblobs_perimeter_6_Nblobs_total_108_a_s_0.2.vertex') as f:
    # If D_eff = D = 0.3246 and a/s = 0.3
    #with open('Cylinder_l_geo_2.12_radius_0.1623_Nblobs_perimeter_6_Nblobs_total_92_a_s_0.3.vertex') as f:
    # If D_eff = D = 0.3246 and a/s = 0.4
    #with open('Cylinder_l_geo_2.12_radius_0.1623_Nblobs_perimeter_6_Nblobs_total_86_a_s_0.4.vertex') as f:    
    # If D_eff = D = 0.3246 and a/s = 0.6
    #with open('Cylinder_l_geo_2.12_radius_0.1623_Nblobs_perimeter_6_Nblobs_total_86_a_s_0.6.vertex') as f:  
    # If D to match resolution 2
    #with open(folder_rods + 'Cylinder_l_geo_2.12_radius_0.14_Nblobs_perimeter_6_Nblobs_total_98.vertex') as f:
    #To match tt_perp of true cylinder with a/s = 0.5    
    # To match tt_perp of true cylinder with a/s = 0.26515
    #with open('Cylinder_l_geo_2.0748_radius_0.15884_Nblobs_perimeter_6_Nblobs_total_96_a_s_0.26515.vertex') as f:

    with open(folder_rods +'Cylinder_l_geo_1.9384_radius_0.1484_Nblobs_perimeter_6_Nblobs_total_86_a_s_0.5.vertex') as f:

      k = 0
      for l in f:
	k = k+1
	if k>1:
	  initial_configuration.append(np.array([float(x) for x in l.strip().split("  ")]))    
      
  elif resolution == 2:
    initial_configuration = []
    # If D_eff = D = 0.3246  and a/s = 0.5
    #with open('Cylinder_l_geo_2.12_radius_0.1623_Nblobs_perimeter_12_Nblobs_total_324.vertex') as f:    
    # If D_eff = D = 0.3246  and a/s = 0.4
    #with open('Cylinder_l_geo_2.12_radius_0.1623_Nblobs_perimeter_12_Nblobs_total_332_a_s_0.4.vertex') as f: 
    # If D_eff = D = 0.3246  and a/s = 0.6
    #with open('Cylinder_l_geo_2.12_radius_0.1623_Nblobs_perimeter_12_Nblobs_total_318_a_s_0.6.vertex') as f:
    # If D_eff = D = 0.3246  and a/s = 0.3
    #with open('Cylinder_l_geo_2.12_radius_0.1623_Nblobs_perimeter_12_Nblobs_total_354_a_s_0.3.vertex') as f:
    # If D = 0.5
    #with open('Cylinder_l_geo_3.2656_radius_0.25_Nblobs_perimeter_12_Nblobs_total_324.vertex') as f:    
    #with open('Cylinder_l_geo_1.6328_radius_0.125_Nblobs_perimeter_12_Nblobs_total_324.vertex') as f:
    #with open('Cylinder_l_geo_0.81639_radius_0.0625_Nblobs_perimeter_12_Nblobs_total_324.vertex') as f:
    # To match tt_perp of true cylinder with a/s = 0.25793
    #with open(folder_rods + 'Cylinder_l_geo_2.1043_radius_0.1611_Nblobs_perimeter_12_Nblobs_total_374_a_s_0.25793.vertex') as f:
    # To match tt_perp of true cylinder with a/s = 0.5
    with open('Cylinder_l_geo_2.0299_radius_0.1554_Nblobs_perimeter_12_Nblobs_total_324_a_s_0.5.vertex') as f:
      k = 0
      for l in f:
	k = k+1
	if k>1:
	  initial_configuration.append(np.array([float(x) for x in l.strip().split("  ")]))    


  return initial_configuration

def get_shell_initial_config(location, orientation):
  '''
  Depends on the resolution chosen
  ''' 
  
  folder_shells = 'Generated_shells/'
 
           
  if resolution == 1:
    initial_configuration = []
    with open(folder_shells +'shell_3d_Nblob_12_radius_0_225.vertex') as f:

      k = 0
      for l in f:
	k = k+1
	if k>1:
	  initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))    
      
  elif resolution == 2:
    initial_configuration = []
    with open(folder_shells +'shell_3d_Nblob_42_radius_0_225.vertex') as f:
      k = 0
      for l in f:
	k = k+1
	if k>1:
	  initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))    
  
  elif resolution == 3:
    initial_configuration = []
    with open(folder_shells +'shell_3d_Nblob_162_radius_0_225.vertex') as f:
      k = 0
      for l in f:
	k = k+1
	if k>1:
	  initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))    

  elif resolution == 4:
    initial_configuration = []
    with open(folder_shells +'shell_3d_Nblob_642_radius_0_225.vertex') as f:
      k = 0
      for l in f:
	k = k+1
	if k>1:
	  initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))    

  elif resolution == 5:
    initial_configuration = []
    with open(folder_shells +'shell_3d_Nblob_2562_radius_0_225.vertex') as f:
      k = 0
      for l in f:
	k = k+1
	if k>1:
	  initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))    


  return initial_configuration

def get_r_vectors(location, orientation, initial_configuration):
  '''
  Rotates the frame config
  ''' 

  rotation_matrix = orientation.rotation_matrix()
  rotated_configuration = []

  for vec in initial_configuration:
    rotated_configuration.append(np.dot(rotation_matrix, vec) + np.array(location))
    
  return rotated_configuration


  
def read_initial_configuration(filename,z):

  k = 0
  initial_location = []
  initial_orientation = []
  with open(filename) as f:      
    for l in f.readlines():
      k = k+1
      if k==3:
	Nrods = int(l)
      if k==4:
	line =  l.strip().split(' ')
        line = [x for x in line if x != '']
	D_raw = float(line[0])
	l_raw = float(line[1])
	AR_raw = l_raw/D_raw
	scale = DIAM_ROD_EXCLU/D_raw
	print "l_raw , D_raw, AR_raw, scale = ", l_raw , D_raw , AR_raw, scale
	print "l_raw*scale = ", l_raw*scale
        #raw_input()      
      if k>6:
        pos_orient =  l.strip().split(' ')
        pos_orient = [x for x in pos_orient if x != '']
        pos = pos_orient[0:2]
        orient = pos_orient[2:4]
        angle = m.atan2(-float(orient[0]),float(orient[1]))
	initial_location.append(np.array([scale*float(pos[0]), scale*float(pos[1]), z]))    
	initial_orientation.append(Quaternion([m.cos(angle/2.), 0.0, 0.0, m.sin(angle/2.)]))  
	#print "angle = ", angle
	#print "initial_orientation = ", initial_orientation
        #print "initial_location = ", initial_location
        #raw_input()
      #if k>10:
        #Nrods = k-7+1
        #print "len(initial_location), Nrods = ", len(initial_location), Nrods
        #break
  return (Nrods,initial_location, initial_orientation)
  
  
def create_initial_configuration_lattice(Nrods,dx,dy,z):

  Nx = int(m.sqrt(float(Nrods)))
  Ny = Nx
  initial_location = []
  initial_orientation = []
  for k in range(Nx):
    for l in range(Ny):
      initial_location.append(np.array([float(k)*dx, float(l)*dy, z]))    
      initial_orientation.append(Quaternion([1.0, 0.0, 0.0, 0.0]))  
	#print "angle = ", angle
  #print "initial_orientation = ", initial_orientation
  print "initial_location = ", initial_location

      #if k>10:
        #Nrods = k-7+1
        #print "len(initial_location), Nrods = ", len(initial_location), Nrods
        #break
  return (initial_location, initial_orientation)



def calc_rot_matrix(r_vectors, location):
  ''' 
  Calculate the matrix R, where the i-th 3x3 block of R gives
  (R_i x) = -1 (r_i cross x). 
  R will be 3N by 3 (18 x 3). The r vectors point from the center
  of the shape to the other vertices.
  '''
  
  Nbody = len(r_vectors)
  Nblobs_per_body = len(r_vectors[0])  
  rot_matrix = np.zeros((3*Nbody*Nblobs_per_body,3))
  for k in range(Nbody):

    for j in range(Nblobs_per_body):

      # Here the cross is relative to the center
      adjusted_r_vector = r_vectors[k][j] - location[k]
      rot_matrix[k*3*Nblobs_per_body + 3*j:k*3*Nblobs_per_body + 3*(j+1),0:3] = \
          np.array([[0.0, adjusted_r_vector[2], -1.*adjusted_r_vector[1]],\
          [-1.*adjusted_r_vector[2], 0.0, adjusted_r_vector[0]],\
          [adjusted_r_vector[1], -1.*adjusted_r_vector[0], 0.0]])
  return rot_matrix


def rod_force_calculator(r_vectors):

  gravity = np.array([0., 0., -1.*sum(M)])

  Nblobs_per_body = len(r_vectors[0])
  Nbody = len(r_vectors)
  repulsion = np.zeros(3*Nbody)
  for k in range(Nbody):
      for i in range(Nblobs_per_body):
        ri = r_vectors[k][i]    
        repulsion[3*k:3*(k+1)] += np.array([0., 0., \
                        (REPULSION_STRENGTH_WALL*((ri[2] - A)/DEBYE_LENGTH_WALL + 1)*
                         np.exp(-1.*(ri[2] - A)/DEBYE_LENGTH_WALL)/ \
                         ((ri[2] - A)**2))])
	# Use a Yukawa potential for blob-blob repulsion
        for l in range(k): 
	  for j in range(Nblobs_per_body):
	    rj = r_vectors[l][j]  
	    rij = rj - ri
	    dist =  np.linalg.norm(rij)
	    
	    if dist <2.*A:
	     print i,j
	     print dist/(2.*A)
	     
	     
	    rep_blobs =  REPULSION_STRENGTH_BLOBS*DIAM_BLOB/dist**2*(1./DEBYE_LENGTH_BLOBS + 1./dist)*\
                         np.exp(-1.*(dist - DIAM_BLOB)/DEBYE_LENGTH_BLOBS)*rij
	    repulsion[3*k:3*(k+1)] += np.array(-rep_blobs)
	    repulsion[3*l:3*(l+1)] += np.array(rep_blobs)
	    
      repulsion[3*k:3*(k+1)] = repulsion[3*k:3*(k+1)] + gravity
      
  
  return repulsion



def rod_torque_calculator(r_vectors,rotation_matrix):
  ''' 
  Calculate torque based on Rod location and orientation.
  location - list of length 1 with location of tracking point of 
             rod.
  orientation - list of length 1 with orientation (as a Quaternion)
                of rod.
  '''

  # Here the big difference with force calculator is that the forces 
  # for all the blobs are stored, in order to compute the torques
  Nblobs_per_body = len(r_vectors[0])
  Nbody = len(r_vectors)
  forces = np.zeros(3*len(r_vectors)*Nblobs_per_body)
  for k in range(Nbody):
      for i in range(Nblobs_per_body):
        ri = r_vectors[k][i]    
        gravity = -1.*M[i]
        forces[3*k*Nblobs_per_body+3*i:3*k*Nblobs_per_body+3*(i+1)] += np.array([0., 0., \
                        (REPULSION_STRENGTH_WALL*((ri[2] - A)/DEBYE_LENGTH_WALL + 1)*
                         np.exp(-1.*(ri[2] - A)/DEBYE_LENGTH_WALL)/ \
                         ((ri[2] - A)**2)) \
			   + gravity])
	# Use a Yukawa potential for blob-blob repulsion
        for l in range(k): 
	  for j in range(Nblobs_per_body):
	    rj = r_vectors[l][j]  
	    rij = rj - ri
	    dist =  np.linalg.norm(rij)
	    rep_blobs =  REPULSION_STRENGTH_BLOBS*DIAM_BLOB/dist**2*(1./DEBYE_LENGTH_BLOBS + 1./dist)*\
                         np.exp(-1.*(dist - DIAM_BLOB)/DEBYE_LENGTH_BLOBS)*rij
	    forces[3*k*Nblobs_per_body+3*i:3*k*Nblobs_per_body+3*(i+1)] += np.array(-rep_blobs)
	    forces[3*l*Nblobs_per_body+3*j:3*l*Nblobs_per_body+3*(j+1)] += np.array(rep_blobs)


  torques = np.zeros(3*Nbody)
  for k in range(Nbody):
      torques[3*k:3*(k+1)] = \
         np.dot(rotation_matrix[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body,:].T,\
                forces[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body])

  return torques


@static_var('normalization_constants', {})
def generate_rod_equilibrium_sample(n_precompute=20000):
  ''' 
  Use accept-reject to generate a sample
  with location and orientation from the Gibbs Boltzmann 
  distribution for the Rod.

  This function is best used to generate many samples at once, as
  there is significant overhead involved in precomputing the
  normalization factor.

  normalization_constants is a dictionary that stores an
  estimated normalization constant for each value of the sum of mass.
  '''
  max_height = KT/sum(M)*12 + A + 4.*DEBYE_LENGTH
  # TODO: Figure this out a better way that includes repulsion.
  # Get a rough upper bound on max height.
  norm_constants = generate_rod_equilibrium_sample.normalization_constants
  if sum(M) in norm_constants.keys():
    normalization_factor = norm_constants[sum(M)]
  else:
    # Estimate normalization constant from random samples
    # and store it.
    max_normalization = 0.
    for k in range(n_precompute):
      theta = np.random.normal(0., 1., 4)
      orientation = Quaternion(theta/np.linalg.norm(theta))
      location = [0., 0., np.random.uniform(A, max_height)]
      accept_prob = rod_gibbs_boltzmann_distribution(location, orientation)
      if accept_prob > max_normalization:
        max_normalization = accept_prob
    
    norm_constants[sum(M)] = max_normalization
    normalization_factor = max_normalization

  while True:
    theta = np.random.normal(0., 1., 4)
    orientation = Quaternion(theta/np.linalg.norm(theta))
    location = [0., 0., np.random.uniform(A, max_height)]
    accept_prob = rod_gibbs_boltzmann_distribution(location, orientation)/(
      2.5*normalization_factor)
    if accept_prob > 1.:
      print 'Accept probability %s is greater than 1' % accept_prob
    
    if np.random.uniform(0., 1.) < accept_prob:
      return [location, orientation]


def rod_gibbs_boltzmann_distribution(location, orientation):
  ''' Return exp(-U/kT) for the given location and orientation.'''
  r_vectors = get_rod_r_vectors(location, orientation)
  # Add gravity to potential.
  for k in range(len(r_vectors)):
    if r_vectors[k][2] < A:
      return 0.0
  U = 0
  for k in range(len(r_vectors)):
    U += M[k]*r_vectors[k][2]
    h = r_vectors[k][2]
    # Add repulsion to potential.
    U += (REPULSION_STRENGTH_WALL*np.exp(-1.*(h -A)/DEBYE_LENGTH_WALL)/
          (h-A))

  return np.exp(-1.*U/KT)


def load_equilibrium_sample(f):
  ''' Load an equilibrium sample from a given file.
  f = file object. should be opened, and parameters should already be read
  through. File is generated by the populate_gibbs_sample.py script.
  '''
  line = f.readline()
  items = line.split(',')
  position = [float(x) for x in items[0:3]]
  orientation = Quaternion([float(x) for x in items[3:7]])
  return [position, orientation]
  

def rod_check_function(locations,orientations,initial_configuration):
  ''' 
  Function called after timesteps to check that the rod
  is in a viable location (not through the wall).
  '''
  r_vectors = []
  for k in range(len(locations)):
    r_vectors.append(get_r_vectors(locations[k], orientations[k],initial_configuration))
    
  Nblobs = len(r_vectors[0])
  for k in range(len(r_vectors)):
    for i in range(Nblobs):
      if r_vectors[k][i][2] < A:
	print r_vectors[k][i][2]
	print A
	raw_input()
        return False
  return True
  

def slip_velocity(r_vectors,locations):
  '''
  Function that returns the slip velocity on each blob.
  '''
  ## Forces
  #return slip_velocity_extensile_rod(r_vectors)
  return slip_velocity_extensile_rod_resolved_distrib(r_vectors,locations)
  # Dipoles of Forces
  #return slip_velocity_extensile_rod_dipoles(locations, orientations)

def slip_velocity_extensile_rod_resolved_distrib(r_vectors,locations):
  '''
  Creates slip for a extensile rod. The slip is along the
  axis pointing to the closest end. We assume the blobs
  are equispaced.
  In this version the slip is constant. 
  '''

  # Slip speed
  speed = -4.0
  if resolution == 0:
   Nblobs_covering_ends = 0
   Nlobs_perimeter = 0
  elif resolution == 1:
   Nblobs_covering_ends = 1
   Nlobs_perimeter = 6 
  elif resolution == 2:
   Nblobs_covering_ends = 9
   Nlobs_perimeter = 12 
  elif resolution == 3:
   Nblobs_covering_ends = 22
   Nlobs_perimeter = 18 
  
  slip = []
  for k in range(len(r_vectors)):
    # Get rod orientation
    number_of_blobs = len(r_vectors[k])
    if resolution>0:
      axis = r_vectors[k][number_of_blobs- 2*Nblobs_covering_ends-2] \
	   - r_vectors[k][Nlobs_perimeter-2]
    else:
      axis = r_vectors[k][number_of_blobs-1] \
	   - r_vectors[k][0]
    length_rod = np.linalg.norm(axis)+2.0*A
    axis = axis / np.sqrt(np.dot(axis, axis))
    if resolution ==0:    
     lower_bound = length_rod/2.0 - 0.7 -0.0001
     #lower_bound = length_rod/2.0 - A -0.0001
    else:
     lower_bound = length_rod/2.0 - 0.7 -0.0001
     
    upper_bound = length_rod/2.0
    #print "lower_bound = ", lower_bound
    #print "upper_bound = ", upper_bound
    #print "axis = ", axis
    #print "length_rod = ", length_rod
    # Create slip  
    slip_blob = []
    for i in range(number_of_blobs):
      if Nblobs_covering_ends>0 and i>=number_of_blobs-2*Nblobs_covering_ends:
	#print "i = ",i
	slip_blob = [0., 0., 0.]
      else:
	dist_COM_along_axis = np.dot(r_vectors[k][i]-locations[k], axis)
	if(dist_COM_along_axis >lower_bound) and \
	  (dist_COM_along_axis <=upper_bound):
	  #print "i, dist_COM_along_axis = ",i, dist_COM_along_axis
	  slip_blob = -speed * axis
	elif(dist_COM_along_axis <-lower_bound) and \
	  (dist_COM_along_axis >=-upper_bound):
	  #print "i, dist_COM_along_axis = ",i, dist_COM_along_axis
	  slip_blob = speed * axis
	else:
	  slip_blob = [0., 0., 0.]
      slip.append(slip_blob[0])
      slip.append(slip_blob[1])
      slip.append(slip_blob[2])
 
  return slip

def slip_velocity_extensile_rod(r_vectors):
  '''
  Creates slip for a extensile rod. The slip is along the
  axis pointing to the closest end. We assume the blobs
  are equispaced.
  In this version the slip is constant. 
  '''

  # Slip speed
  #speed = -1.0
  # random slip
  
  
  slip = []
  for k in range(len(r_vectors)):
    # Get rod orientation
    number_of_blobs = len(r_vectors[k])
    axis = r_vectors[k][number_of_blobs-1] - r_vectors[k][0]
    # random slip direction 
    #axis = 2.0*np.random.random(3) - 1.0
    axis = axis / np.sqrt(np.dot(axis, axis))

    # Create slip
  
    slip_blob = []
    for i in range(number_of_blobs):
      #speed = 2.0*np.random.random() - 1.0
      if(i < number_of_blobs / 2):
	slip_blob = speed * axis
      elif(i > (number_of_blobs / 2)):
	slip_blob = -speed * axis
      elif((i == (number_of_blobs / 2)) and (number_of_blobs % 2 == 0)):
	slip_blob = -speed * axis
      else:
	slip_blob = [0., 0., 0.]
      slip.append(slip_blob[0])
      slip.append(slip_blob[1])
      slip.append(slip_blob[2])


  return slip


def resistance_blobs(r_vectors):
  '''
  This function compute the resistance matrix at the blob level
  '''
  
  # Blobs mobility
  r_vec_for_mob = []
  for k in range(len(r_vectors)):
    r_vec_for_mob += r_vectors[k]

  mobility = mb.boosted_single_wall_fluid_mobility(r_vec_for_mob, ETA, A)

  # Resistance blobs
  resistance = np.linalg.inv(mobility)

  return resistance

def mobility_blobs(locations, orientations):
  '''
  This function compute the mobility matrix at the blob level
  '''
  
  # Blobs mobility
  r_vec_for_mob = []
  for k in range(len(locations)):
    r_vec_for_mob += get_rod_r_vectors(locations[k], orientations[k])



  mobility = mb.boosted_single_wall_fluid_mobility(r_vec_for_mob, ETA, A)

  return mobility
  
  
def matrices_for_GMRES_iteration(locations, orientations, initial_configuration):
  '''
  
  '''
  
  # Get vectors
  r_vectors = []
  for k in range(len(locations)):
    r_vectors.append(get_r_vectors(locations[k], orientations[k],initial_configuration))
    
  Nbody = len(r_vectors)
  Nblobs_per_body = len(r_vectors[0])
  Ncomp_blobs = 3*Nbody*Nblobs_per_body

  rotation_matrix = calc_rot_matrix(r_vectors, locations)
  
  ### INSTEAD OF DEFINING J_tot = None, 
  ### I SHOULD SPECIFY ITS SIZE TO AVOID CONCATENATE SO MUCH
  #J_tot = np.zeros((3*Nblobs_per_body,6*Nbody))
  self_mobility_body = np.zeros((6*Nbody,6))
  mobility_blobs_each_body = np.zeros((Ncomp_blobs,3*Nblobs_per_body))
  chol_mobility_blobs_each_body = np.zeros((Ncomp_blobs,3*Nblobs_per_body))
  
  J_tot = None
  J = np.concatenate([np.identity(3) for \
                     _ in range(Nblobs_per_body)])  
  for k in range(Nbody):
    ### AVOID TOO MUCH CONCATENATION!

    J_rot_combined = np.concatenate([J, rotation_matrix[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body,:]], axis=1)

    if J_tot is None:
        J_tot = J_rot_combined
    else:
        J_tot=np.concatenate([J_tot, J_rot_combined], axis=1)
        
    ## Mobilility with wall correction
    mobility = mb.boosted_single_wall_fluid_mobility(r_vectors[k], ETA, A)
    # Mobilility without wall
    #mobility = mb.boosted_infinite_fluid_mobility(r_vectors[k], ETA, A)

    chol_mobility_blobs_each_body[k*3*Nblobs_per_body:(k+1)*3*Nblobs_per_body,:] \
                                  = np.linalg.cholesky(mobility)   
     
    resistance = np.linalg.inv(mobility)
    
    ## TO UNCOMMENT OR USE A FLAG FOR CYLINDER == pinv
    #self_mobility_body[6*k:6*(k+1),:] = np.linalg.pinv(np.dot(J_rot_combined.T, \
                                        #np.dot(resistance, \
                                               #J_rot_combined)))
    self_mobility_body[6*k:6*(k+1),:] = np.linalg.inv(np.dot(J_rot_combined.T, \
                                        np.dot(resistance, \
                                               J_rot_combined)))


			     

    

  return (J_tot,self_mobility_body,chol_mobility_blobs_each_body,r_vectors, rotation_matrix)
  
  

def matrices_for_direct_iteration(locations, orientations, initial_configuration):
  '''
  
  '''
  
  # Get vectors
  r_vectors = []
  for k in range(len(locations)):
    r_vectors.append(get_r_vectors(locations[k], orientations[k],initial_configuration))
    
  rotation_matrix = calc_rot_matrix(r_vectors, locations)
  

  return (r_vectors, rotation_matrix)
  
  
def mobility_vector_prod(r_vectors, vector):
  '''
  This function compute the mobility matrix at the blob level
  '''
  
  # Blobs mobility
  r_vec_for_mob = []
  for k in range(len(r_vectors)):
    r_vec_for_mob += r_vectors[k]



  #res1 = mb.boosted_mobility_vector_product(r_vec_for_mob, ETA, A,vector)
  #res2 = mb.single_wall_mobility_times_force_pycuda(r_vec_for_mob, vector, ETA, A)
  
  #print "np.linalg.norm(C++-CUDA) = "
  #print np.linalg.norm(res1-res2)
  #raw_input()
  
  res = mb.single_wall_mobility_times_force_pycuda(r_vec_for_mob, vector, ETA, A)

  #res = mb.single_wall_mobility_times_force_pycuda_single(r_vec_for_mob, vector, ETA, A)

  return res


def K_matrix_T_vector_prod(r_vectors, rotation_matrix, lambda_slip):
  '''
  Compute the K^T matrix,  this matrix transport the information from the level of
  describtion of the blobs to the level of describtion of the 
  body.
  Then perform the operation: K^*\cdot lambda_slip
  '''

  # K matrix
  Nbody = len(r_vectors)
  Nblobs_per_body = len(r_vectors[0])
  J_trans = np.concatenate([np.identity(3) for \
                     _ in range(Nblobs_per_body)])

  force_slip = np.zeros(6*Nbody)
  
  for k in range(Nbody):
       #force_slip[3*k:3*(k+1)] =\
	      #np.dot(J_tot[:,6*k:6*k+3].T,\
		     #lambda_slip[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body])
       force_slip[3*k:3*(k+1)] =\
	      np.dot(J_trans.T,\
		     lambda_slip[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body])	     
       #force_slip[3*Nbody+3*k:3*Nbody+3*(k+1)] =\
	      #np.dot(J_tot[:,6*k+3:6*(k+1)].T,\
		     #lambda_slip[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body])
       force_slip[3*Nbody+3*k:3*Nbody+3*(k+1)] =\
	      np.dot(rotation_matrix[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body,0:3].T,\
		     lambda_slip[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body])
  return force_slip

def K_matrix_vector_prod(r_vectors, rotation_matrix, vel_body):
  '''
  Compute the K matrix,  this matrix transport the information from the level of
  describtion of the body to the level of describtion of the 
  blobs.
  Then perform the operation: K\cdot vel_body
  '''
  

  Nbody = len(r_vectors)
  Nblobs_per_body = len(r_vectors[0])
  J_trans = np.concatenate([np.identity(3) for \
                     _ in range(Nblobs_per_body)])

  
  vel_blobs = np.zeros(3*Nbody*Nblobs_per_body)
  for k in range(Nbody):
       vel_blobs[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body] =\
	      np.dot(J_trans, vel_body[3*k:3*(k+1)]) \
	      + np.dot(rotation_matrix[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body,:],\
	               vel_body[3*Nbody+3*k:3*Nbody+3*(k+1)])
  return vel_blobs
  
  
def K_matrix(r_vectors, rotation_matrix):
  '''
  Compute the K matrix,  this matrix transport the information from the level of
  describtion of the body to the level of describtion of the 
  blobs.
  '''
  

  Nbodies = len(r_vectors)
  Nblobs_per_body = len(r_vectors[0])
  J_tot = np.zeros((3*Nblobs_per_body*Nbodies,6*Nbodies))
  J = np.concatenate([np.identity(3) for \
                     _ in range(Nblobs_per_body)])
  for k in range(Nbodies):
    J_tot[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body,3*k:3*(k+1)] = J
    J_tot[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body,3*Nbodies+3*k:3*Nbodies+3*(k+1)] = \
          rotation_matrix[3*k*Nblobs_per_body:3*(k+1)*Nblobs_per_body,:]

  return J_tot
  
  

  

def linear_operator_rigid(vector, r_vectors, rotation_matrix, Nbody, Ncomp_blobs):
  '''
  Return the action of the linear operator of the rigid body on vector v
  '''

  
  res = np.zeros(Ncomp_blobs + 6*Nbody)
  
  res[0:Ncomp_blobs] = \
           mobility_vector_prod(r_vectors,vector[0:Ncomp_blobs]) \
           - K_matrix_vector_prod(r_vectors,rotation_matrix,vector[Ncomp_blobs:Ncomp_blobs+ 6*Nbody] ) 
           
  res[Ncomp_blobs:Ncomp_blobs+ 6*Nbody] = -K_matrix_T_vector_prod(r_vectors, rotation_matrix, vector[0:Ncomp_blobs])        

  return res


def preconditioner_gmres(vector, K_matrix, mob_chol_blobs, self_mob_body):
  '''
  Preconditioner which solves for each body separately
  '''
 
  Nbody = self_mob_body.shape[0]/6
  Nblobs_per_body = mob_chol_blobs.shape[1]/3
  Ncomp_blobs = 3*Nbody*Nblobs_per_body
  res = np.zeros(Ncomp_blobs + 6*Nbody)


  for k in range(Nbody):
    # 1)Solve M*Lambda_tilde = slip

    mobility_chol = mob_chol_blobs[k*3*Nblobs_per_body:(k+1)*3*Nblobs_per_body,0:3*Nblobs_per_body]

    lambda_tilde = -sla.cho_solve((mobility_chol,True),\
                                 vector[k*3*Nblobs_per_body:(k+1)*3*Nblobs_per_body])
    
    # 2) Compute rigid body velocity Y_tilde
    total_mobility = self_mob_body[6*k:6*(k+1),0:6]
    my_K_matrix = K_matrix[0:3*Nblobs_per_body,6*k:6*(k+1)]
   
      
    my_force_torque = np.concatenate([vector[Ncomp_blobs+k*3:Ncomp_blobs+(k+1)*3],\
                                     vector[Ncomp_blobs+3*Nbody+k*3:Ncomp_blobs+3*Nbody+(k+1)*3] ])
 
    
    product = -np.dot( total_mobility, my_force_torque\
                                     - np.dot(my_K_matrix.T,lambda_tilde) )
                                     
    res[Ncomp_blobs+k*3:Ncomp_blobs+(k+1)*3]=product[0:3]
    res[Ncomp_blobs+3*Nbody+k*3:Ncomp_blobs+3*Nbody+(k+1)*3]=product[3:6]
    
    # Only for python 2.7
    #product_compatible = np.hstack(product)
    #res[Ncomp_blobs+k*3:Ncomp_blobs+(k+1)*3]=product_compatible[0:3]
    #res[Ncomp_blobs+3*Nbody+k*3:Ncomp_blobs+3*Nbody+(k+1)*3]=product_compatible[3:6]
            

    # 3) Solve M*Lamda_tilde_2 = (slip + K*Y_tilde)
    res[k*3*Nblobs_per_body:(k+1)*3*Nblobs_per_body] = \
                          sla.cho_solve( (mobility_chol,True),\
                               vector[k*3*Nblobs_per_body:(k+1)*3*Nblobs_per_body] \
                             + np.dot(my_K_matrix,product) )
    # Only for python 2.7                       
    #res[k*3*Nblobs_per_body:(k+1)*3*Nblobs_per_body] = \
                          #np.hstack(sla.cho_solve( (mobility_chol,True),\
                               #vector[k*3*Nblobs_per_body:(k+1)*3*Nblobs_per_body] \
                             #+ np.dot(J_rot_combined,product) ))                      


  
  return res


  
if __name__ == '__main__':
  # Get command line arguments.
  parser = argparse.ArgumentParser(description='Run Simulation of Rod '
                                   'particle with Fixman, EM, and RFD '
                                   'schemes, and save trajectory.  Rod '
                                   'is affected by gravity and repulsed from '
                                   'the wall gently.')
  parser.add_argument('-dt', dest='dt', type=float,
                      help='Timestep to use for runs.')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps to take for runs.')
  parser.add_argument('-gfactor', dest='gravity_factor', type=float, default=1.0,
                      help='Factor to increase gravity by.')
  parser.add_argument('-scheme', dest='scheme', type=str, default='RFD',
                      help='Numerical Scheme to use: RFD, FIXMAN, or EM.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      default='',
                      help='Optional name added to the end of the '
                      'data file.  Useful for multiple runs. '
                      'To analyze multiple runs and compute MSD, you must '
                      'specify this, and it must end with "-#" '
                      ' for # starting at 1 and increasing successively. e.g. '
                      'heavy-masses-1, heavy-masses-2, heavy-masses-3 etc.')
  parser.add_argument('--profile', dest='profile', type=bool, default=False,
                      help='True or False: Profile this run or not.')

  args=parser.parse_args()
  if args.profile:
    pr = cProfile.Profile()
    pr.enable()

  # Increase gravity
  M = np.array(M)*args.gravity_factor

  # Get command line parameters
  dt = args.dt
  n_steps = args.n_steps
  print_increment = max(int(n_steps/20.), 1)

  # Set up logging.
  if(True):
    log_filename = './logs/rod-dt-%f-N-%d-scheme-%s-g-%s-%s.log' % (dt, n_steps, args.scheme, args.gravity_factor, args.data_name)
    flog = open(log_filename, 'w')
    #progress_logger = logging.getLogger('Progress Logger')
    #progress_logger.setLevel(logging.INFO)
    ## Add the log message handler to the logger
    #logging.basicConfig(filename=log_filename, level=logging.INFO, filemode='w')
    #sl = StreamToLogger(progress_logger, logging.INFO)
    #sys.stdout = sl
    #sl = StreamToLogger(progress_logger, logging.ERROR)
    #sys.stderr = sl
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, flog)

  # Gather parameters to save
  params = {'A': A, 'ETA': ETA, 'M': M,
            'REPULSION_STRENGTH_WALL': REPULSION_STRENGTH_WALL,
            'DEBYE_LENGTH_WALL': DEBYE_LENGTH_WALL, 'dt': dt, 'n_steps': n_steps,
            'gfactor': args.gravity_factor, 'scheme': args.scheme,
            'KT': KT}

  print "Parameters for this run are: ", params

  
  #z = 0.3246*2.0
  z = (0.225+A) + float(n_steps-1)*0.01
  z = 0.711
  z = 0.3553875
  print "dz = ", z
  
  #folder_packings = 'Generated_packings/'
  ## Script to run the various integrators on the quaternion.
  #if resolution == 0:
    #filename = folder_packings + 'PackSuperellipses_Nr_10000_Nb_21_AR_7_543_phi_0_1.dat'
  #elif resolution == 1:
    #filename = folder_packings +  'PackSuperellipses_Nr_10000_Nb_98_AR_5_381_phi_0_1.dat'
  #elif resolution == 2:
    #filename = folder_packings +  'PackSuperellipses_Nr_10_Nb_324_AR_5_394_phi_0_1.dat'
  
  #print " Open file : ", filename
  #(Nrods,initial_location,initial_orientation)  = \
          #read_initial_configuration(filename,z )
  
  #dx = 5.06
  #dy = 1.2984
  #Nrods = 9
  #(initial_location,initial_orientation) = create_initial_configuration_lattice(Nrods,dx,dy,z)

  
  
  
  initial_location = [np.array([0.0e0, 0.47385e0*0.5, z]),\
                      np.array([0.47385*1.25, 0.0e0, z]),]
		    
  initial_orientation = [Quaternion([1.0, 0., 0., 0.]),\
                         Quaternion([1.0, 0., 0., 0.]),]
		       
		       
  #initial_location = [[0.0e0, 0.0e0, 0.7e0],\
                      #[5.0e0, 5.0e0, 0.7e0],\
                      #[10.0e0, 10.0e0, 0.7e0],]
		    
  #initial_orientation = [Quaternion([1., 0., 0., 0.]),\
                         #Quaternion([0.707106781186548, 0., 0., 0.707106781186548]),\
                         ##Quaternion([1.0, 0., 0., 0.0]),\
                         #Quaternion([1, 0., 0., 0.])]

  Nrods =len(initial_location)
   
  Nblobs = Nrods*Nblobs_per_rod
  
  
  #initial_configuration = get_rod_initial_config(initial_location,initial_orientation)
  initial_configuration = get_shell_initial_config(initial_location,initial_orientation)

 
  print "Nrods = ", Nrods
  print "Nblobs = ", Nblobs
  
  # If we don't include BM, choice_solver = GMRES (1)
  # If we include BM, choice_solver = direct (2)
  if KT==0.0:
    choice_solver = 1
  else:
    choice_solver = 2  

  quaternion_integrator = QuaternionIntegratorGMRES(rod_mobility,
                                               initial_orientation, 
                                               rod_torque_calculator, 
                                               has_location=True,
                                               initial_location=initial_location,
                                               force_calculator=rod_force_calculator,
                                               slip_velocity=slip_velocity,
                                               resistance_blobs=resistance_blobs,
                                               mobility_blobs=mobility_blobs,
                                               mobility_vector_prod=mobility_vector_prod,
                                               #linear_operator = linear_operator_rigid,
                                               #get_vectors = get_rod_r_vectors,
                                               blob_vel = K_matrix_vector_prod,
                                               force_slip = K_matrix_T_vector_prod)
  quaternion_integrator.kT = KT
  quaternion_integrator.A = A
  quaternion_integrator.Nrods = Nrods
  quaternion_integrator.Nblobs = Nblobs
  quaternion_integrator.check_function = rod_check_function
  quaternion_integrator.linear_operator = linear_operator_rigid 
  quaternion_integrator.precond = preconditioner_gmres  
  quaternion_integrator.get_vectors = get_r_vectors
  quaternion_integrator.matrices_for_GMRES_ite = matrices_for_GMRES_iteration
  quaternion_integrator.matrices_for_direct_ite = matrices_for_direct_iteration
  quaternion_integrator.first_guess = np.zeros(Nblobs*3 + Nrods*6)
  quaternion_integrator.initial_config = initial_configuration
  quaternion_integrator.solver = choice_solver

  trajectory = [[], []]

  if len(args.data_name) > 0:
    def generate_trajectory_name(scheme):
      trajectory_dat_name = 'rod-trajectory-dt-%g-N-%d-scheme-%s-g-%s-%s.txt' % (dt, n_steps, scheme, args.gravity_factor, args.data_name)
      return trajectory_dat_name
  else:
    def generate_trajectory_name(scheme):
      trajectory_dat_name = 'rod-trajectory-dt-%g-N-%d-scheme-%s-g-%s.txt' % (dt, n_steps, scheme, args.gravity_factor)
      return trajectory_dat_name

  data_file = os.path.join(
    DATA_DIR, 'data', generate_trajectory_name(args.scheme))
  write_trajectory_to_txt(data_file, trajectory, params)

  # First check that the directory exists.  If not, create it.
  dir_name = os.path.dirname(data_file)
  if not os.path.isdir(dir_name):
     os.mkdir(dir_name)

  # Write data to file, parameters first then trajectory.
  with open(data_file, 'w', 1) as f:
    f.write('Parameters:\n')
    for key, value in params.items():
      f.writelines(['%s: %s \n' % (key, value)])
    f.write('Trajectory data:\n')
    f.write('Location, Orientation, Velocity, Rotation:\n')

     # Current location/orientation is the state at which velocities
     # and rotations are computed
    current_location = initial_location
    current_orientation = initial_orientation

    start_time = time.time()  
    for k in range(n_steps):
      
      
      # Fixman step and bin result.
      if args.scheme == 'FIXMAN':
        quaternion_integrator.fixman_time_step(dt)
      elif args.scheme == 'RFD':
        quaternion_integrator.rfd_time_step(dt)
      elif args.scheme == 'EM':
        # EM step and bin result.
        quaternion_integrator.additive_em_time_step(dt)
      else:
        raise Exception('scheme must be one of: RFD, FIXMAN, EM.')
      
      

      for l in range(len(initial_location)):
	#veltot = quaternion_integrator.veltot[l]
	#omegatot = quaternion_integrator.omegatot[l]
	my_orientation = current_orientation[l].entries
	mob_coeffs = quaternion_integrator.mob_coeff[l]
	#f.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s \n' % (
	f.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s \n' % (
	  current_location[l][0], current_location[l][1], current_location[l][2], \
	  my_orientation[0], my_orientation[1], my_orientation[2], my_orientation[3],\
	  mob_coeffs[0][0], mob_coeffs[0][4],\
	  mob_coeffs[1][1], mob_coeffs[1][3],\
	  mob_coeffs[2][2], mob_coeffs[3][3],\
	  mob_coeffs[4][4], mob_coeffs[5][5] ))
	  #current_location[l][0], current_location[l][1], current_location[l][2], \
	  #my_orientation[0], my_orientation[1], my_orientation[2], my_orientation[3],\
	  #veltot[0], veltot[1], veltot[2],\
	  #omegatot[0], omegatot[1], omegatot[2]))
       
       # Update positions and orientations
      current_location = []
      current_orientation = []
      for l in range(len(initial_location)):
        current_location.append(quaternion_integrator.location[l])
        current_orientation.append(quaternion_integrator.orientation[l])
       

      if k % print_increment == 0:
        elapsed_time = time.time() - start_time
        print 'At step %s out of %s' % (k, n_steps)
        log_time_progress(elapsed_time, k, n_steps)
      

  elapsed_time = time.time() - start_time
  if(False):
    if elapsed_time > 60:
      progress_logger.info('Finished timestepping. Total Time: %.2f minutes.' % (float(elapsed_time)/60.))
    else:
      progress_logger.info('Finished timestepping. Total Time: %.2f seconds.' %  float(elapsed_time))     
    progress_logger.info('Integrator Rejection rate: %s' % (float(quaternion_integrator.rejections) / float(quaternion_integrator.rejections + n_steps)))


  if elapsed_time > 60:
    print 'Finished timestepping. Total Time: %.2f minutes.' % (float(elapsed_time)/60.)
  else:
    print 'Finished timestepping. Total Time: %.2f seconds.' %  float(elapsed_time)
  print 'Integrator Rejection rate: %s' % (float(quaternion_integrator.rejections) / float(quaternion_integrator.rejections + n_steps))


  #flog.close()



  if args.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

 
