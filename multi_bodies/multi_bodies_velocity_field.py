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


from config_local import DATA_DIR
from mobility import mobility as mb
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from utils import log_time_progress
from utils import static_var
from utils import StreamToLogger
from utils import write_trajectory_to_txt

# Make sure figures folder exists
if not os.path.isdir(os.path.join(os.getcwd(), 'figures')):
  os.mkdir(os.path.join(os.getcwd(), 'figures'))
# Make sure logs folder exists
if not os.path.isdir(os.path.join(os.getcwd(), 'logs')):
  os.mkdir(os.path.join(os.getcwd(), 'logs'))


# Control th resolution of the rod
resolution = 1
resolution_sphere = 1
print "resolution = ", resolution
print "resolution_sphere = ", resolution_sphere

# Parameters.  Units are um, s, mg.
# TO ADD AN IF STATEMENT TO CHOSE BETWEEN SHELLS AND CYLINDERS
#if resolution == 0:
  
 ##A = 0.1623 # for a cylinder with the lowest resolution
 ##Nblobs_per_rod = 18
 
 #A = 0.225 # A = 0.225 Matches The Mobility Coefficients of the 12 blob model
 #Nblobs_per_rod = 12
 #if A == 0.1623:
  #if resolution_sphere == 0: 
    #A_blob_shell = A
  #elif resolution_sphere == 1: 
    #A_blob_shell = 0.086377621721174 # 12 blobs
  #elif resolution_sphere == 2:
    #A_blob_shell = 0.044897702647680 # 42 blobs
  #elif resolution_sphere == 3: 
    #A_blob_shell = 0.022665551644763 # 162 blobs
    
 #elif A == 0.225: 
  #if resolution_sphere == 0: 
    #A_blob_shell = A
  #elif resolution_sphere == 1: 
    #A_blob_shell = 0.1183 # 12 blobs
  #elif resolution_sphere == 2:
    #A_blob_shell = 0.061484985366573 # 42 blobs
  #elif resolution_sphere == 3: 
    #A_blob_shell = 0.031039252100254 # 162 blobs
  #elif resolution_sphere == 4: 
    #A_blob_shell = 0.015556856646539 # 642 blobs

#elif resolution == 1:
 ##A = 0.064920000000000 # For 6 blobs along the rod perimeter and diam rod = 0.3246
 #A = 0.081150000000000 # If we set Rg = D/2
 #Nblobs_per_rod = 80
 #if resolution_sphere == 0: 
  #A_blob_shell = A
 #elif resolution_sphere == 1: 
  #A_blob_shell = 0.042663079748468 # 12 blobs
 #elif resolution_sphere == 2:
  #A_blob_shell = 0.022175578821255 # 42 blobs
 #elif resolution_sphere == 3:
  #A_blob_shell = 0.011194824448650 # 162 blobs
#elif resolution == 2:
 ##A = 0.037193179428176 # For 12 blobs along the rod perimeter and diam rod = 0.3246
 #A = 0.042006331020139 # If we set Rg = D/2
 #Nblobs_per_rod = 330
 #if resolution_sphere == 0: 
  #A_blob_shell = A
 #elif resolution_sphere == 1: 
  #A_blob_shell = 0.022084035123262 # 12 blobs
 #elif resolution_sphere == 2:
  #A_blob_shell = 0.011478924270225 # 42 blobs
 #elif resolution_sphere == 3:
  #A_blob_shell = 0.005794867547780 # 162 blobs
#elif resolution == 3:
 ##A = 0.025931610759192 # For 18 blobs along the rod perimeter and diam rod = 0.3246
 #A = 0.028183099235343 # If we set Rg = D/2
 #Nblobs_per_rod = 746
 #if resolution_sphere == 0: 
  #A_blob_shell = A
 #elif resolution_sphere == 1: 
  #A_blob_shell = 0.0148167321039612 # 12 blobs
 #elif resolution_sphere == 2:
  #A_blob_shell = 0.007701497702040 # 42 blobs
 #elif resolution_sphere == 3:
  #A_blob_shell = 0.003887921729618 # 162 blobs

# BLOB RADII FOR SHELL WITH Rg = 0.225
if resolution == 1: 
  A = 0.1183 # 12 blobs
  Nblobs_per_rod = 86
elif resolution== 2:
  A= 0.061484985366573 # 42 blobs
  Nblobs_per_rod = 42
elif resolution== 3: 
  A = 0.031039252100254 # 162 blobs
  Nblobs_per_rod = 162
  if resolution_sphere == 0: 
   A_blob_shell = A
  elif resolution_sphere == 1: 
    A_blob_shell = 0.016318300526013 # 12 blobs
  elif resolution_sphere == 2:
    A_blob_shell = 0.008481988681476 # 42 blobs
  elif resolution_sphere == 3:
    A_blob_shell = 0.004281934421191 # 162 blobs
elif resolution == 4: 
  A = 0.015556856646539 # 642 blobs
  Nblobs_per_rod = 642
  if resolution_sphere == 0: 
   A_blob_shell = A
  elif resolution_sphere == 1: 
    A_blob_shell = 0.008178723545863 # 12 blobs
  elif resolution_sphere == 2:
    A_blob_shell = 0.004251168216589 # 42 blobs
  elif resolution_sphere == 3:
    A_blob_shell = 0.002146103254846 # 162 blobs
elif resolution == 5: 
  A = 0.007783085222670 # 642 blobs
  Nblobs_per_rod = 2562




DIAM = 2.0*A  # Diameter of the blobs
ETA = 1  # Water = 1e-3 Pa s =1e-3 kg/(m s) = 1e-3 mg/(um s)

# density of particle = 20.03 g/cm^3 = 2.003e-11 g/um^3.  
# Volume is ~0.2 um^3. 
# weigth = Volume*density = 0.2*2e-11 = 4e-12g = 4e-9mg
TOTAL_MASS = 0.0*4e-9*(9.8*1.e6)
M = [TOTAL_MASS/Nblobs_per_rod for _ in range(Nblobs_per_rod)]
#KT = 1.3806488e-23 m^2*kg/(s^2*K) = 1.3806488e-5 um^2*mg/(s^2*K)
KT = 0.0*1.3806488e-5# 300.*1.3806488e-5  # T = 300K

# Made these up somewhat arbitrarily
## The following values are for lower bound  = l/2 - 0.7- 0.001
# To have a stable rod at z = 0.3266 with resolution =1, 
#                Nblobs = 80 and mass = 4e-9 and DEBYE = 0.5A
#REPULSION_STRENGTH_WALL = 1.8024e-2* 300.*1.3806488e-5 #KT
# To have a stable rod at z = 0.3266 with resolution =2, 
#                Nblobs = 330 and mass = 4e-9 and DEBYE = 0.5A
#REPULSION_STRENGTH_WALL = 1.898e-1* 300.*1.3806488e-5 #KT
# To have a stable rod at z = 0.3266 with resolution =0, and A = 0.1623
#                Nblobs = 18 and mass = 4e-9 and DEBYE = 0.5A
#REPULSION_STRENGTH_WALL = 3.7902e-2* 300.*1.3806488e-5 #KT
# To have a stable rod at z = 0.3266 with resolution =0, and A = 0.225
#                Nblobs = 12 and mass = 4e-9 and DEBYE = 0.5A
#REPULSION_STRENGTH_WALL = 1.19718e-2* 300.*1.3806488e-5 #KT

## The following values are for lower bound  = l/2 - 0.8- 0.001
# To have a stable rod at z = 0.3266 with resolution =0, and A = 0.225
#                Nblobs = 12 and mass = 4e-9 and DEBYE = 0.5A
#REPULSION_STRENGTH_WALL = 1.19045e-2* 300.*1.3806488e-5 #KT

REPULSION_STRENGTH_WALL = 0.0*1.19718e-2* 300.*1.3806488e-5 #KT

DEBYE_LENGTH_WALL = 0.5*A # 0.1*A
#DEBYE_LENGTH_WALL = 0.5*0.1643 # 0.1*A

# Made these up somewhat arbitrarily
REPULSION_STRENGTH_BLOBS =0.0* 0.02 * 300.*1.3806488e-5 #KT #2.0* 300.*1.3806488e-5
DEBYE_LENGTH_BLOBS = 0.005*A

# Made these up somewhat arbitrarily
ATTRACTION_STRENGTH_BLOBS = 0.*2. * 300.*1.3806488e-5 #KT
ATTRACTION_LENGTH_BLOBS = 0.1*A


def rod_mobility(locations, orientations):
  ''' 
  Calculate the force and torque mobility for the
  rod.  Here location is the cross point.
  '''
  num_blobs_per_rod = len(get_rod_r_vectors(locations[0], orientations[0]))
  r_vectors = np.empty([len(locations), num_blobs_per_rod, 3])
  for k in range(len(locations)):
    r_vectors[k] = get_rod_r_vectors(locations[k], orientations[k])
  return force_and_torque_rod_mobility(r_vectors, locations)

def sphere_mobility(locations, orientations):
  ''' 
  Calculate the force and torque mobility for the
  rod.  Here location is the cross point.
  '''
  r_vectors = []
  for k in range(len(locations)):
    r_vectors.append(get_sphere_r_vectors(locations[k], orientations[k]))

  return force_and_torque_rod_mobility(r_vectors, locations)


def rod_mobility_at_arbitrary_point(locations, orientations, point):
  '''
  Calculate the force and torque mobility for the
  rod.  Here location is the cross point, but point is 
  some other arbitrary point.  

  The returned mobility is the (force, torque) -> (velocity, angular
  velocity) mobility for forces applied to <point>, torques about
  <point>, and the resulting velocity of <point>
  '''
  r_vectors = get_rod_r_vectors(locations[0], orientations[0])
  return force_and_torque_rod_mobility(r_vectors, point)


def force_and_torque_rod_mobility(r_vectors, location):
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
  # r_vec_for_mob = []
  # for k in range(len(r_vectors)):
  # r_vec_for_mob += r_vectors[k]
  # print 'r_vectors', r_vectors

  mobility = mb.boosted_single_wall_fluid_mobility(r_vectors, ETA, A)

  # K matrix
  rotation_matrix = calc_rot_matrix(r_vectors, location)
  Nbody = len(r_vectors)
  J_tot = None
  for k in range(Nbody):
    J = np.concatenate([np.identity(3) for \
                     _ in range(len(r_vectors[k]))])

    J_rot_combined = np.concatenate([J, rotation_matrix[3*k*len(r_vectors[k]):3*(k+1)*len(r_vectors[k]),:]], axis=1)

    if J_tot is None:
        J_tot = J_rot_combined
    else:
        J_tot=np.concatenate([J_tot, J_rot_combined], axis=1)

  mob_inv = np.linalg.inv(mobility)

  total_resistance = np.zeros((6*len(r_vectors),6*len(r_vectors)))
  
  # ONLY BUILD UPPER TRIANGULAR PART OF R AND ASSIGN THE REST
  for k in range(Nbody):
    len_k = len(r_vectors[k])
    for j in range(k,Nbody):
        len_j = len(r_vectors[j]) 
  
        # VF BLOCK
        total_resistance[3*k:3*(k+1),3*j:3*(j+1)] =\
             np.dot(np.dot(J_tot[:,6*k:6*k+3].T,\
                    mob_inv[3*k*len_k:3*(k+1)*len_k,3*j*len_j:3*(j+1)*len_j]),\
                    J_tot[:,6*j:6*j+3])
        if j>k:
	  total_resistance[3*j:3*(j+1),3*k:3*(k+1)] = \
                    total_resistance[3*k:3*(k+1),3*j:3*(j+1)].T

        # VT BLOCK
        total_resistance[3*k:3*(k+1),3*(Nbody+j):3*(Nbody+j+1)] =\
             np.dot(np.dot(J_tot[:,6*k:6*k+3].T,\
                    mob_inv[3*k*len_k:3*(k+1)*len_k,3*j*len_j:3*(j+1)*len_j]),\
                    J_tot[:,6*j+3:6*(j+1)])
        if j>k:
            total_resistance[3*j:3*(j+1),3*(Nbody+k):3*(Nbody+k+1)] =\
                     np.dot(np.dot(J_tot[:,6*j:6*j+3].T,\
                    mob_inv[3*j*len_j:3*(j+1)*len_j,3*k*len_k:3*(k+1)*len_k]),\
                    J_tot[:,6*k+3:6*(k+1)])

        # WT BLOCK
        total_resistance[3*(Nbody+k):3*(Nbody+k+1),3*(Nbody+j):3*(Nbody+j+1)] =\
             np.dot(np.dot(J_tot[:,6*k+3:6*(k+1)].T,\
                    mob_inv[3*k*len_k:3*(k+1)*len_k,3*j*len_j:3*(j+1)*len_j]),\
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

  print "location  = ", location
  print "6*6 mobility = "
  for i in range(6):
    for j in range(6):
      if abs(total_mobility[i][j])>1e-10:
	print i,j
	print total_mobility[i][j]
  #raw_input()
  return total_mobility


def calculate_rod_coh(location, orientation):
  ''' return CoH of the rod, given location and orientation.
  This uses the 15 blob rod.'''
  dist = 1.07489
  r_vectors = get_rod_r_vectors(location, orientation)
  arm_1 = r_vectors[0] - location
  arm_2 = r_vectors[14] - location
  coh_point = (location + arm_1*np.sin(np.pi/4.)*dist/2.1 
               + arm_2*np.sin(np.pi/4.)*dist/2.1)
  return coh_point

def calculate_rod_cod(location, orientation):
  ''' return CoD (CoM) of the rod, given location and orientation.
  This uses the 15 blob rod.'''
  dist = 0.96087
  r_vectors = get_rod_r_vectors(location, orientation)
  arm_1 = r_vectors[0] - location
  arm_2 = r_vectors[14] - location
  coh_point = (location + arm_1*np.sin(np.pi/4.)*dist/2.1 
               + arm_2*np.sin(np.pi/4.)*dist/2.1)
  return coh_point



def get_rod_r_vectors(location, orientation):
  '''
  Depends on the resolution chosen
  if resolution == 0:
    Get the vectors of the 10 blobs used to discretize the cylinder.
 
           0-0-0-O-O-O-O-O-O-O-O
          10 9 8 7 6 5 4 3 2 1 0
  if resolution == 1:
    Get the vectors of the blobs from a file where the surface of 
    the cylinder with blobs
  The location is the location of the middle of the cylinder
  The effective hydrodynamic length of the cylinder
  is Leff=(distance_first_last_blobs) + 2*radius_blob
  
  #'''
  #resolution = 0
  #if resolution == 0:
    #initial_configuration = [np.array([0.941288, 0., 0.]),
			    #np.array([0.732112888889, 0., 0.]),
			    #np.array([0.522937777778, 0., 0.]),
			    #np.array([0.313762666667, 0., 0.]),
			    #np.array([0.104587555556, 0., 0.]),
			    #np.array([-0.104587555556, 0., 0.]),
			    #np.array([-0.313762666667, 0., 0.]),
			    #np.array([-0.522937777778, 0., 0.]),
			    #np.array([-0.732112888889, 0., 0.]),
			    #np.array([-0.941288, 0., 0.])]
  
  #
  
  folder_rods = 'Generated_rods/'
  if resolution == 0:
    initial_configuration = []
    #with open('Cylinder_l_2.2_radius_0.1623_Nblobs_perimeter_1_Nblobs_total_18.vertex') as f:
    with open(folder_rods + 'Cylinder_l_2.2_radius_0.225_Nblobs_perimeter_1_Nblobs_total_12.vertex') as f:        
    #with open('Cylinder_l_3.246_radius_0.1623_Nblobs_perimeter_1_Nblobs_total_28.vertex') as f:        
    #with open('Cylinder_l_1.2984_radius_0.1623_Nblobs_perimeter_1_Nblobs_total_10.vertex') as f:
      k = 0
      for l in f:
	k = k+1
	if k>1:
	  initial_configuration.append(np.array([float(x) for x in l.strip().split("  ")]))    

  elif resolution == 1:
    initial_configuration = []
    # If D_eff = D/(1+a_blob/4)
    #with open('Cylinder_l_2.2_radius_0.1623_Nblobs_perimeter_6_Nblobs_total_98.vertex') as f:
    # If D_eff = D
    with open(folder_rods + 'Cylinder_l_geo_2.12_radius_0.1623_Nblobs_perimeter_6_Nblobs_total_86.vertex') as f:
    #with open('Cylinder_l_2.2_radius_0.1623_Nblobs_perimeter_6_Nblobs_total_158.vertex') as f:
    #with open('Cylinder_l_3.246_radius_0.1623_Nblobs_perimeter_6_Nblobs_total_122.vertex') as f:
    # Radius geo to get radius hydro  = 0.1623
    #with open('Cylinder_l_3.246_radius_0.14151_Nblobs_perimeter_6_Nblobs_total_134.vertex') as f:    
    #with open('Cylinder_l_1.2984_radius_0.1623_Nblobs_perimeter_6_Nblobs_total_50.vertex') as f:
    # Radius geo to get radius hydro  = 0.1623
    #with open('Cylinder_l_1.2984_radius_0.14151_Nblobs_perimeter_6_Nblobs_total_56.vertex') as f:    

      k = 0
      for l in f:
	k = k+1
	if k>1:
	  initial_configuration.append(np.array([float(x) for x in l.strip().split("  ")]))    
      
  elif resolution == 2:
    initial_configuration = []
    # If D_eff = D/(1+a_blob/4)
    #with open('Cylinder_l_2.2_radius_0.1623_Nblobs_perimeter_12_Nblobs_total_366.vertex') as f:
    # If D_eff = D    
    # with open(folder_rods + 'Cylinder_l_2.2_radius_0.1623_Nblobs_perimeter_12_Nblobs_total_330.vertex') as f:    
    #with open('Cylinder_l_3.246_radius_0.1623_Nblobs_perimeter_12_Nblobs_total_474.vertex') as f:    
    with open('Cylinder_l_1.2984_radius_0.1623_Nblobs_perimeter_12_Nblobs_total_198.vertex') as f:
      k = 0
      for l in f:
	k = k+1
	if k>1:
	  initial_configuration.append(np.array([float(x) for x in l.strip().split("  ")]))    
	  

  elif resolution == 3:
    initial_configuration = []
    # If D_eff = D/(1+a_blob/4)    
    #with open('Cylinder_l_2.2_radius_0.1623_Nblobs_perimeter_18_Nblobs_total_800.vertex') as f:   
    # If D_eff = D
    with open(folder_rods + 'Cylinder_l_2.2_radius_0.1623_Nblobs_perimeter_18_Nblobs_total_746.vertex') as f:    
    #with open('Cylinder_l_3.246_radius_0.1623_Nblobs_perimeter_18_Nblobs_total_1070.vertex') as f:       
    #with open('Cylinder_l_1.2984_radius_0.1623_Nblobs_perimeter_18_Nblobs_total_458.vertex') as f:
      k = 0
      for l in f:
	k = k+1
	if k>1:
	  initial_configuration.append(np.array([float(x) for x in l.strip().split("  ")]))    
      

  elif resolution == 4:
    initial_configuration = []
    with open(folder_rods + 'Cylinder_l_1.2984_radius_0.1623_Nblobs_perimeter_36_Nblobs_total_1808.vertex') as f:
      k = 0
      for l in f:
	k = k+1
	if k>1:
	  initial_configuration.append(np.array([float(x) for x in l.strip().split("  ")]))    
      

  rotation_matrix = orientation.rotation_matrix()
  rotated_configuration = []

  for vec in initial_configuration:
    rotated_configuration.append(np.dot(rotation_matrix, vec) + np.array(location))
    

  return rotated_configuration


def get_sphere_for_rod_r_vectors(location):
  '''Get the vectors of the  blobs used to discretize a sphere.
 
 
  The location is the location of the center of the sphere 
  Initial configuration is in the
  The radius of the blob A is chosen so that the distance between two
  neighbouring blobs is 2A
  The radius of the sphere is 1
  '''
  
  folder_shells = 'Generated_shells/'
  initial_configuration = []

  if resolution == 0:
    if A == 0.1643:
      if resolution_sphere == 1:
	with open(folder_shells + 'shell_3d_Nblob_12_radius_0_1643.vertex') as f:
	  k = 0
	  for l in f:
	    k = k+1
	    if k>1:
	      initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
      elif resolution_sphere == 2:
	with open(folder_shells + 'shell_3d_Nblob_42_radius_0_1643.vertex') as f:
	  k = 0
	  for l in f:
	    k = k+1
	    if k>1:
	      initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
      elif resolution_sphere == 3:
	with open(folder_shells + 'shell_3d_Nblob_162_radius_0_1643.vertex') as f:
	  k = 0
	  for l in f:
	    k = k+1
	    if k>1:
	      initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
    elif A == 0.225:
      if resolution_sphere == 1:
	with open(folder_shells + 'shell_3d_Nblob_12_radius_0_225.vertex') as f:
	  k = 0
	  for l in f:
	    k = k+1
	    if k>1:
	      initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
      elif resolution_sphere == 2:
	with open(folder_shells + 'shell_3d_Nblob_42_radius_0_225.vertex') as f:
	  k = 0
	  for l in f:
	    k = k+1
	    if k>1:
	      initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
      elif resolution_sphere == 3:
	with open(folder_shells + 'shell_3d_Nblob_162_radius_0_225.vertex') as f:
	  k = 0
	  for l in f:
	    k = k+1
	    if k>1:
	      initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
      elif resolution_sphere == 4:
	with open(folder_shells + 'shell_3d_Nblob_642_radius_0_225.vertex') as f:
	  k = 0
	  for l in f:
	    k = k+1
	    if k>1:
	      initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
	      
  elif resolution == 1:
    if resolution_sphere == 1:
      with open(folder_shells + 'shell_3d_Nblob_12_radius_0_081.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
    elif resolution_sphere == 2:
      with open(folder_shells + 'shell_3d_Nblob_42_radius_0_081.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
    elif resolution_sphere == 3:
      with open(folder_shells + 'shell_3d_Nblob_162_radius_0_081.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
  elif resolution == 2:
    if resolution_sphere == 1:
      with open(folder_shells + 'shell_3d_Nblob_12_radius_0_042.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
    elif resolution_sphere == 2:
      with open(folder_shells + 'shell_3d_Nblob_42_radius_0_042.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
    elif resolution_sphere == 3:
      with open(folder_shells + 'shell_3d_Nblob_162_radius_0_042.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
  elif resolution == 3:
    if resolution_sphere == 1:
      with open(folder_shells + 'shell_3d_Nblob_12_radius_0_028.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
    elif resolution_sphere == 2:
      with open(folder_shells + 'shell_3d_Nblob_42_radius_0_028.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
    elif resolution_sphere == 3:
      with open(folder_shells + 'shell_3d_Nblob_162_radius_0_028.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
	  
  translated_configuration = []
  for vec in initial_configuration:
    translated_configuration.append(vec + np.array(location))

  
  #print "rotated_configuration = ", rotated_configuration
  #raw_input()

  return translated_configuration


def get_sphere_r_vectors(location, orientation):
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


  rotation_matrix = orientation.rotation_matrix()
  rotated_configuration = []

  for vec in initial_configuration:
    rotated_configuration.append(np.dot(rotation_matrix, vec) + np.array(location))
    

  return rotated_configuration


def get_sphere_for_sphere_r_vectors(location):
  '''Get the vectors of the  blobs used to discretize a sphere.
 
 
  The location is the location of the center of the sphere 
  Initial configuration is in the
  The radius of the blob A is chosen so that the distance between two
  neighbouring blobs is 2A
  The radius of the sphere is 1
  '''
  
  folder_shells = 'Generated_shells/'
  initial_configuration = []
  if resolution == 3:
    if resolution_sphere == 1:
      with open(folder_shells + 'shell_3d_Nblob_12_radius_0_031.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
    elif resolution_sphere == 2:
      with open(folder_shells + 'shell_3d_Nblob_42_radius_0_031.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
    elif resolution_sphere == 3:
      with open(folder_shells + 'shell_3d_Nblob_162_radius_0_031.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
	  
  elif resolution == 4:
    if resolution_sphere == 1:
      with open(folder_shells + 'shell_3d_Nblob_12_radius_0_01556.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
    elif resolution_sphere == 2:
      with open(folder_shells + 'shell_3d_Nblob_42_radius_0_01556.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
    elif resolution_sphere == 3:
      with open(folder_shells + 'shell_3d_Nblob_162_radius_0_01556.vertex') as f:
	k = 0
	for l in f:
	  k = k+1
	  if k>1:
	    initial_configuration.append(np.array([float(x) for x in l.strip().split("\t")]))
	  
  translated_configuration = []
  for vec in initial_configuration:
    translated_configuration.append(vec + np.array(location))

  
  #print "rotated_configuration = ", rotated_configuration
  #raw_input()

  return translated_configuration


def calc_rot_matrix(r_vectors, location):
  ''' 
  Calculate the matrix R, where the i-th 3x3 block of R gives
  (R_i x) = -1 (r_i cross x). 
  R will be 3N by 3 (18 x 3). The r vectors point from the center
  of the shape to the other vertices.
  '''

  rot_matrix = None
  for k in range(len(r_vectors)):

    for j in range(len(r_vectors[k])):

      # Here the cross is relative to the center
      adjusted_r_vector = r_vectors[k][j] - location[k]
      block = np.array( \
          [[0.0, adjusted_r_vector[2], -1.*adjusted_r_vector[1]],\
          [-1.*adjusted_r_vector[2], 0.0, adjusted_r_vector[0]],\
          [adjusted_r_vector[1], -1.*adjusted_r_vector[0], 0.0]])
      if rot_matrix is None:
        rot_matrix = block
      else:
        rot_matrix = np.concatenate([rot_matrix, block], axis=0)
  return rot_matrix


def rod_force_calculator(location, orientation):

  gravity = np.array([0., 0., -1.*sum(M)])
  r_vectors = []
  for k in range(len(location)):
     r_vectors.append(get_rod_r_vectors(location[k], orientation[k]))
  
  repulsion = np.zeros(3*len(r_vectors))
  for k in range(len(r_vectors)):
      for i in range(len(r_vectors[k])):
        ri = r_vectors[k][i]    
        repulsion[3*k:3*(k+1)] += np.array([0., 0., \
                        (REPULSION_STRENGTH_WALL*((ri[2] - A)/DEBYE_LENGTH_WALL + 1)*
                         np.exp(-1.*(ri[2] - A)/DEBYE_LENGTH_WALL)/ \
                         ((ri[2] - A)**2))])
	# Use a Yukawa potential for blob-blob repulsion
        for l in range(k): 
	  for j in range(len(r_vectors[l])):
	    rj = r_vectors[l][j]  
	    rij = rj - ri
	    dist =  np.linalg.norm(rij)
	    rep_blobs =  REPULSION_STRENGTH_BLOBS*DIAM/dist**2*(1./DEBYE_LENGTH_BLOBS + 1./dist)*\
                         np.exp(-1.*(dist - DIAM)/DEBYE_LENGTH_BLOBS)*rij
            
	    electric_blobs =  ATTRACTION_STRENGTH_BLOBS*DIAM/dist**2*(1./ATTRACTION_LENGTH_BLOBS + 1./dist)*\
                         np.exp(-1.*(dist - DIAM)/ATTRACTION_LENGTH_BLOBS)*rij
            # Same material --> repulsion
            if (i<3 or i>=len(r_vectors[k])-3) and (j<3 or j>=len(r_vectors[l])-3):
	      repulsion[3*k:3*(k+1)] += np.array(-rep_blobs-electric_blobs)
	      repulsion[3*l:3*(l+1)] += np.array(rep_blobs+electric_blobs)
	      #print "i,j,rij,-electric_blobs", i,j
	      #print rij
	      #print -electric_blobs
	      #raw_input()
	    # Same material --> repulsion
	    elif(i>=3 and i<len(r_vectors[l])-3) and (j>=3 and j<len(r_vectors[l])-3):
	      repulsion[3*k:3*(k+1)] += np.array(-rep_blobs-electric_blobs)
	      repulsion[3*l:3*(l+1)] += np.array(rep_blobs+electric_blobs)
	      #print "i,j,rij,-electric_blobs", i,j
	      #print rij
	      #print -electric_blobs
	      #raw_input()
	    # Different material --> attraction
	    else:
	      repulsion[3*k:3*(k+1)] += np.array(-rep_blobs+electric_blobs)
	      repulsion[3*l:3*(l+1)] += np.array(rep_blobs-electric_blobs)
	      #print "i,j,rij,+electric_blobs", i,j
	      #print rij
	      #print +electric_blobs
	      #raw_input()
	    
      repulsion[3*k:3*(k+1)] = repulsion[3*k:3*(k+1)] + gravity
      
  #print "repulsion = ", repulsion
  #raw_input()
  return repulsion



def rod_torque_calculator(location, orientation):
  ''' 
  Calculate torque based on Rod location and orientation.
  location - list of length 1 with location of tracking point of 
             rod.
  orientation - list of length 1 with orientation (as a Quaternion)
                of rod.
  '''
  # r_vectors = []
  # for k in range(len(location)):
  # r_vectors.append(get_rod_r_vectors(location[k], orientation[k]))
  num_blobs_per_rod = len(get_rod_r_vectors(location[0], orientation[0]))
  r_vectors = np.empty([len(location), num_blobs_per_rod, 3])
  for k in range(len(location)):
    r_vectors[k] = get_rod_r_vectors(location[k], orientation[k])
  # Here the big difference with force calculator is that the forces 
  # for all the blobs are stored, in order to compute the torques
  Nblobs = r_vectors.size / 3
  forces = np.zeros(r_vectors.size)

  # print r_vectors[0]

  for k in range(len(r_vectors)):
      for i in range(Nblobs):
        # print k, i, Nblobs
        ri = r_vectors[k][i]
        gravity = -1.*M[i]
        forces[3*k*Nblobs+3*i:3*k*Nblobs+3*(i+1)] += np.array([0., 0., \
                        (REPULSION_STRENGTH_WALL*((ri[2] - A)/DEBYE_LENGTH_WALL + 1)*
                         np.exp(-1.*(ri[2] - A)/DEBYE_LENGTH_WALL)/ \
                         ((ri[2] - A)**2)) \
			   + gravity])
	# Use a Yukawa potential for blob-blob repulsion
        for l in range(k): 
	  for j in range(len(r_vectors[l])):
	    rj = r_vectors[l][j]  
	    rij = rj - ri
	    dist =  np.linalg.norm(rij)
	    rep_blobs =  REPULSION_STRENGTH_BLOBS*DIAM/dist**2*(1./DEBYE_LENGTH_BLOBS + 1./dist)*\
                         np.exp(-1.*(dist - DIAM)/DEBYE_LENGTH_BLOBS)*rij
            electric_blobs = ATTRACTION_STRENGTH_BLOBS*DIAM/dist**2*(1./ATTRACTION_LENGTH_BLOBS + 1./dist)*\
                         np.exp(-1.*(dist - DIAM)/ATTRACTION_LENGTH_BLOBS)*rij
            # Same material --> repulsion
            if (i<3 or i>=len(r_vectors[k])-3) and (j<3 or j>=len(r_vectors[l])-3):
	      forces[3*k*Nblobs+3*i:3*k*Nblobs+3*(i+1)] += np.array(-rep_blobs-electric_blobs)
	      forces[3*l*Nblobs+3*j:3*l*Nblobs+3*(j+1)] += np.array(rep_blobs+electric_blobs)
	    # Same material --> repulsion
	    elif(i>=3 and i<len(r_vectors[l])-3) and (j>=3 and j<len(r_vectors[l])-3):
	      forces[3*k*Nblobs+3*i:3*k*Nblobs+3*(i+1)] += np.array(-rep_blobs-electric_blobs)
	      forces[3*l*Nblobs+3*j:3*l*Nblobs+3*(j+1)] += np.array(rep_blobs+electric_blobs)
	    # Different material --> attraction
	    else:
	      forces[3*k*Nblobs+3*i:3*k*Nblobs+3*(i+1)] += np.array(-rep_blobs+electric_blobs)
	      forces[3*l*Nblobs+3*j:3*l*Nblobs+3*(j+1)] += np.array(rep_blobs-electric_blobs)
	      
  R = calc_rot_matrix(r_vectors, location)
  torques = np.zeros(3*len(r_vectors))
  for k in range(len(r_vectors)):
      torques[3*k:3*(k+1)] = \
         np.dot(R[3*k*Nblobs:3*(k+1)*Nblobs,:].T,\
                forces[3*k*Nblobs:3*(k+1)*Nblobs])
	      
  #print "torques = ", torques
  #raw_input()
  return torques




def sphere_force_calculator(location, orientation):

  gravity = np.array([0., 0., -1.*sum(M)])
  r_vectors = []
  for k in range(len(location)):
     r_vectors.append(get_sphere_r_vectors(location[k], orientation[k]))
  
  repulsion = np.zeros(3*len(r_vectors))
  for k in range(len(r_vectors)):
      for i in range(len(r_vectors[k])):
        ri = r_vectors[k][i]    
        repulsion[3*k:3*(k+1)] += np.array([0., 0., \
                        (REPULSION_STRENGTH_WALL*((ri[2] - A)/DEBYE_LENGTH_WALL + 1)*
                         np.exp(-1.*(ri[2] - A)/DEBYE_LENGTH_WALL)/ \
                         ((ri[2] - A)**2))])
	# Use a Yukawa potential for blob-blob repulsion
        for l in range(k): 
	  for j in range(len(r_vectors[l])):
	    rj = r_vectors[l][j]  
	    rij = rj - ri
	    dist =  np.linalg.norm(rij)
	    rep_blobs =  REPULSION_STRENGTH_BLOBS*DIAM/dist**2*(1./DEBYE_LENGTH_BLOBS + 1./dist)*\
                         np.exp(-1.*(dist - DIAM)/DEBYE_LENGTH_BLOBS)*rij
            
	    electric_blobs =  ATTRACTION_STRENGTH_BLOBS*DIAM/dist**2*(1./ATTRACTION_LENGTH_BLOBS + 1./dist)*\
                         np.exp(-1.*(dist - DIAM)/ATTRACTION_LENGTH_BLOBS)*rij
            # Same material --> repulsion
            if (i<3 or i>=len(r_vectors[k])-3) and (j<3 or j>=len(r_vectors[l])-3):
	      repulsion[3*k:3*(k+1)] += np.array(-rep_blobs-electric_blobs)
	      repulsion[3*l:3*(l+1)] += np.array(rep_blobs+electric_blobs)
	      #print "i,j,rij,-electric_blobs", i,j
	      #print rij
	      #print -electric_blobs
	      #raw_input()
	    # Same material --> repulsion
	    elif(i>=3 and i<len(r_vectors[l])-3) and (j>=3 and j<len(r_vectors[l])-3):
	      repulsion[3*k:3*(k+1)] += np.array(-rep_blobs-electric_blobs)
	      repulsion[3*l:3*(l+1)] += np.array(rep_blobs+electric_blobs)
	      #print "i,j,rij,-electric_blobs", i,j
	      #print rij
	      #print -electric_blobs
	      #raw_input()
	    # Different material --> attraction
	    else:
	      repulsion[3*k:3*(k+1)] += np.array(-rep_blobs+electric_blobs)
	      repulsion[3*l:3*(l+1)] += np.array(rep_blobs-electric_blobs)
	      #print "i,j,rij,+electric_blobs", i,j
	      #print rij
	      #print +electric_blobs
	      #raw_input()
	    
      repulsion[3*k:3*(k+1)] = repulsion[3*k:3*(k+1)] + gravity
      
  #print "repulsion = ", repulsion
  #raw_input()
  return repulsion



def sphere_torque_calculator(location, orientation):
  ''' 
  Calculate torque based on Rod location and orientation.
  location - list of length 1 with location of tracking point of 
             rod.
  orientation - list of length 1 with orientation (as a Quaternion)
                of rod.
  '''
  r_vectors = []
  for k in range(len(location)):
     r_vectors.append(get_sphere_r_vectors(location[k], orientation[k]))
  # Here the big difference with force calculator is that the forces 
  # for all the blobs are stored, in order to compute the torques
  Nblobs = len(r_vectors[0])
  forces = np.zeros(3*len(r_vectors)*Nblobs)
  for k in range(len(r_vectors)):
      for i in range(Nblobs):
        ri = r_vectors[k][i]    
        gravity = -1.*M[i]
        forces[3*k*Nblobs+3*i:3*k*Nblobs+3*(i+1)] += np.array([0., 0., \
                        (REPULSION_STRENGTH_WALL*((ri[2] - A)/DEBYE_LENGTH_WALL + 1)*
                         np.exp(-1.*(ri[2] - A)/DEBYE_LENGTH_WALL)/ \
                         ((ri[2] - A)**2)) \
			   + gravity])
	# Use a Yukawa potential for blob-blob repulsion
        for l in range(k): 
	  for j in range(len(r_vectors[l])):
	    rj = r_vectors[l][j]  
	    rij = rj - ri
	    dist =  np.linalg.norm(rij)
	    rep_blobs =  REPULSION_STRENGTH_BLOBS*DIAM/dist**2*(1./DEBYE_LENGTH_BLOBS + 1./dist)*\
                         np.exp(-1.*(dist - DIAM)/DEBYE_LENGTH_BLOBS)*rij
            electric_blobs = ATTRACTION_STRENGTH_BLOBS*DIAM/dist**2*(1./ATTRACTION_LENGTH_BLOBS + 1./dist)*\
                         np.exp(-1.*(dist - DIAM)/ATTRACTION_LENGTH_BLOBS)*rij
            # Same material --> repulsion
            if (i<3 or i>=len(r_vectors[k])-3) and (j<3 or j>=len(r_vectors[l])-3):
	      forces[3*k*Nblobs+3*i:3*k*Nblobs+3*(i+1)] += np.array(-rep_blobs-electric_blobs)
	      forces[3*l*Nblobs+3*j:3*l*Nblobs+3*(j+1)] += np.array(rep_blobs+electric_blobs)
	    # Same material --> repulsion
	    elif(i>=3 and i<len(r_vectors[l])-3) and (j>=3 and j<len(r_vectors[l])-3):
	      forces[3*k*Nblobs+3*i:3*k*Nblobs+3*(i+1)] += np.array(-rep_blobs-electric_blobs)
	      forces[3*l*Nblobs+3*j:3*l*Nblobs+3*(j+1)] += np.array(rep_blobs+electric_blobs)
	    # Different material --> attraction
	    else:
	      forces[3*k*Nblobs+3*i:3*k*Nblobs+3*(i+1)] += np.array(-rep_blobs+electric_blobs)
	      forces[3*l*Nblobs+3*j:3*l*Nblobs+3*(j+1)] += np.array(rep_blobs-electric_blobs)
	      
  R = calc_rot_matrix(r_vectors, location)
  torques = np.zeros(3*len(r_vectors))
  for k in range(len(r_vectors)):
      torques[3*k:3*(k+1)] = \
         np.dot(R[3*k*Nblobs:3*(k+1)*Nblobs,:].T,\
                forces[3*k*Nblobs:3*(k+1)*Nblobs])
	      
  #print "torques = ", torques
  #raw_input()
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
  

def rod_check_function(location, orientation):
  ''' 
  Function called after timesteps to check that the rod
  is in a viable location (not through the wall).
  '''
  r_vectors = []
  for k in range(len(location)):
     r_vectors.append(get_rod_r_vectors(location[k], orientation[k]))
  Nblobs = len(r_vectors[0])
  for k in range(len(r_vectors)):
    for i in range(Nblobs):
      if r_vectors[k][i][2] < A:
	print r_vectors[k][i][2]
	print A
	# raw_input()
        return False
  return True
  

def slip_velocity(locations, orientations):
  '''
  Function that returns the slip velocity on each blob.
  '''
  ## Forces
  #return slip_velocity_extensile_rod_distrib(locations, orientations)
  return slip_velocity_extensile_rod_resolved_distrib(locations, orientations)
  # Dipoles of Forces
  #return slip_velocity_extensile_rod_dipoles(locations, orientations)


def slip_velocity_sphere(locations, orientations):
  '''
  Function that returns the slip velocity on each blob.
  '''
  ## Forces
  #return slip_velocity_extensile_rod_distrib(locations, orientations)
  return slip_velocity_extensile_rod_resolved_distrib_sphere(locations, orientations)
  # Dipoles of Forces
  #return slip_velocity_extensile_rod_dipoles(locations, orientations)




def slip_velocity_extensile_rod_resolved_distrib(locations, orientations):
  '''
  Creates slip for a extensile rod. The slip is along the
  axis pointing to the closest end. We assume the blobs
  are equispaced.
  In this version the slip is constant. 
  '''

  # Slip speed
  speed = -0.0
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
  for k in range(len(locations)):
    # Get rod orientation
    r_vectors = get_rod_r_vectors(locations[k], orientations[k])
    number_of_blobs = len(r_vectors)

    if resolution>0:
      axis = r_vectors[number_of_blobs- 2*Nblobs_covering_ends-2] \
	   - r_vectors[Nlobs_perimeter-2]
    else:
      axis = r_vectors[number_of_blobs-1] \
	   - r_vectors[0]
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
	dist_COM_along_axis = np.dot(r_vectors[i]-locations[k], axis)
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



def slip_velocity_extensile_rod_resolved_distrib_sphere(locations, orientations):
  '''
  Creates slip for a extensile rod. The slip is along the
  axis pointing to the closest end. We assume the blobs
  are equispaced.
  In this version the slip is constant. 
  '''

  # Slip speed
  speed = -0.0
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
  for k in range(len(locations)):
    # Get rod orientation
    r_vectors = get_sphere_r_vectors(locations[k], orientations[k])
    number_of_blobs = len(r_vectors)

    if resolution>0:
      axis = r_vectors[number_of_blobs- 2*Nblobs_covering_ends-2] \
	   - r_vectors[Nlobs_perimeter-2]
    else:
      axis = r_vectors[number_of_blobs-1] \
	   - r_vectors[0]
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
	dist_COM_along_axis = np.dot(r_vectors[i]-locations[k], axis)
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

def slip_velocity_extensile_rod_distrib(locations, orientations):
  '''
  Creates slip for a extensile rod. The slip is along the
  axis pointing to the closest end. We assume the blobs
  are equispaced.
  In this version the slip is constant. 
  '''

  # Slip speed
  speed = -1.0
  
  
  slip = []
  for k in range(len(locations)):
    # Get rod orientation
    r_vectors = get_rod_r_vectors(locations[k], orientations[k])
    number_of_blobs = len(r_vectors)
    Nb_slip = 3
    axis = r_vectors[number_of_blobs-1] - r_vectors[0]
    axis = axis / np.sqrt(np.dot(axis, axis))

    # Create slip
  
    slip_blob = []
    for i in range(number_of_blobs):
      if(i < number_of_blobs / 2 and i <Nb_slip):
	slip_blob = speed * axis
      elif(i > number_of_blobs / 2 and i >=number_of_blobs-Nb_slip):
	slip_blob = -speed * axis
      elif((i == (number_of_blobs / 2)) and (number_of_blobs % 2 == 0) and Nb_slip>=number_of_blobs/2):
	slip_blob = -speed * axis
      else:
	slip_blob = [0., 0., 0.]
      #print "i,slip_blob = ", i,slip_blob
      #raw_input()
      slip.append(slip_blob[0])
      slip.append(slip_blob[1])
      slip.append(slip_blob[2])
      
  

  return slip

#def slip_velocity_extensile_rod_distrib_varies(locations, orientations):
  #'''
  #Creates slip for a extensile rod. The slip is along the
  #axis pointing to the closest end. We assume the blobs
  #are equispaced.
  #In this version the slip is constant. 
  #'''

  ## Slip speed
  #speed = -1.0
  
  
  #slip = []
  #for k in range(len(locations)):
    ## Get rod orientation
    #r_vectors = get_rod_r_vectors(locations[k], orientations[k])
    #number_of_blobs = len(r_vectors)
    #Nb_slip = 10
    #axis = r_vectors[number_of_blobs-1] - r_vectors[0]
    #axis = axis / np.sqrt(np.dot(axis, axis))

    ## Create slip
  
    #slip_blob = []
    #for i in range(number_of_blobs):
      #if(i < number_of_blobs / 2 and i <Nb_slip):
	#slip_blob = float(i+1)*speed * axis
      #elif(i > number_of_blobs / 2 and i >=number_of_blobs-Nb_slip):
	#slip_blob = float(number_of_blobs-i)*-speed * axis
      #elif((i == (number_of_blobs / 2)) and (number_of_blobs % 2 == 0) and Nb_slip>=number_of_blobs/2):
	#slip_blob = float(number_of_blobs-i)*-speed * axis
      #else:
	#slip_blob = [0., 0., 0.]
      ##print "i,slip_blob = ", i,slip_blob
      ##raw_input()
      #slip.append(slip_blob[0])
      #slip.append(slip_blob[1])
      #slip.append(slip_blob[2])
      
  

  #return slip


def slip_velocity_extensile_rod_dipoles(locations, orientations):
  '''
  Creates slip for a extensile rod. The slip is along the
  axis pointing to the closest end. We assume the blobs
  are equispaced.
  In this version the slip is constant. 
  '''

  # Slip speed
  speed = -1.0
  
  slip = []
  for k in range(len(locations)):
    # Get rod orientation
    r_vectors = get_rod_r_vectors(locations[k], orientations[k])
    number_of_blobs = len(r_vectors)
    axis = r_vectors[number_of_blobs-1] - r_vectors[0]
    axis = axis / np.sqrt(np.dot(axis, axis))

    # Create slip
  
    slip_blob = []
    for i in range(number_of_blobs):
      slip_blob = speed*(-1.)**(i) * axis

      slip.append(slip_blob[0])
      slip.append(slip_blob[1])
      slip.append(slip_blob[2])


  return slip


def resistance_blobs(locations, orientations):
  '''
  This function compute the resistance matrix at the blob level
  '''
  
  # Blobs mobility
  # r_vec_for_mob = []
  # for k in range(len(locations)):
  # r_vec_for_mob += get_rod_r_vectors(locations[k], orientations[k])
  num_blobs_per_rod = len(get_rod_r_vectors(locations[0], orientations[0]))
  r_vectors = np.empty([len(locations), num_blobs_per_rod, 3])
  for k in range(len(locations)):
    r_vectors[k] = get_rod_r_vectors(locations[k], orientations[k])

  mobility = mb.boosted_single_wall_fluid_mobility(r_vectors, ETA, A)

  # Resistance blobs
  resistance = np.linalg.inv(mobility)

  return resistance

def resistance_blobs_sphere(locations, orientations):
  '''
  This function compute the resistance matrix at the blob level
  '''
  
  # Blobs mobility
  r_vec_for_mob = []
  for k in range(len(locations)):
    r_vec_for_mob += get_sphere_r_vectors(locations[k], orientations[k])

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



def Compute_force_slip(locations, orientations, lambda_slip):
  '''
  Compute the K matrix,  this matrix transport the information from the level of
  describtion of the body to the level of describtion of the 
  blobs.
  Then perform the operation: K^*\cdot lambda_slip
  '''
  
  # Get vectors
  r_vectors = []
  for k in range(len(locations)):
    r_vectors.append(get_rod_r_vectors(locations[k], orientations[k]))

  # K matrix
  rotation_matrix = calc_rot_matrix(r_vectors, locations)
  Nbody = len(r_vectors)
  J_tot = None
  for k in range(Nbody):
    J = np.concatenate([np.identity(3) for \
                     _ in range(len(r_vectors[k]))])

    J_rot_combined = np.concatenate([J, rotation_matrix[3*k*len(r_vectors[k]):3*(k+1)*len(r_vectors[k]),:]], axis=1)

    if J_tot is None:
        J_tot = J_rot_combined
    else:
        J_tot=np.concatenate([J_tot, J_rot_combined], axis=1)

  force_slip = np.zeros(6*len(r_vectors))
  for k in range(Nbody):
       len_k = len(r_vectors[k])
       force_slip[3*k:3*(k+1)] =\
	      -np.dot(J_tot[:,6*k:6*k+3].T,\
		     lambda_slip[3*k*len_k:3*(k+1)*len_k])
       force_slip[3*Nbody+3*k:3*Nbody+3*(k+1)] =\
	      -np.dot(J_tot[:,6*k+3:6*(k+1)].T,\
		     lambda_slip[3*k*len_k:3*(k+1)*len_k])
  return force_slip


def K_matrix(locations, orientations):
  '''
  Compute the K matrix,  this matrix transport the information from the level of
  describtion of the body to the level of describtion of the 
  blobs.
  '''
  
  # Get vectors
  r_vectors = []
  for k in range(len(locations)):
    r_vectors.append(get_rod_r_vectors(locations[k], orientations[k]))

  # K matrix
  rotation_matrix = calc_rot_matrix(r_vectors, locations)
  Nbodies = len(r_vectors)
  Nblobs = len(r_vectors[0])
  J_tot = np.zeros((3*Nblobs*Nbodies,6*Nbodies))
  for k in range(Nbodies):
    J = np.concatenate([np.identity(3) for \
                     _ in range(Nblobs)])
    J_tot[3*k*Nblobs:3*(k+1)*Nblobs,3*k:3*(k+1)] = J
    J_tot[3*k*Nblobs:3*(k+1)*Nblobs,3*Nbodies+3*k:3*Nbodies+3*(k+1)] = \
          rotation_matrix[3*k*Nblobs:3*(k+1)*Nblobs,:]

  return J_tot

def K_matrix_sphere(locations, orientations):
  '''
  Compute the K matrix,  this matrix transport the information from the level of
  describtion of the body to the level of describtion of the 
  blobs.
  '''
  
  # Get vectors
  r_vectors = []
  for k in range(len(locations)):
    r_vectors.append(get_sphere_r_vectors(locations[k], orientations[k]))

  # K matrix
  rotation_matrix = calc_rot_matrix(r_vectors, locations)
  Nbodies = len(r_vectors)
  Nblobs = len(r_vectors[0])
  J_tot = np.zeros((3*Nblobs*Nbodies,6*Nbodies))
  for k in range(Nbodies):
    J = np.concatenate([np.identity(3) for \
                     _ in range(Nblobs)])
    J_tot[3*k*Nblobs:3*(k+1)*Nblobs,3*k:3*(k+1)] = J
    J_tot[3*k*Nblobs:3*(k+1)*Nblobs,3*Nbodies+3*k:3*Nbodies+3*(k+1)] = \
          rotation_matrix[3*k*Nblobs:3*(k+1)*Nblobs,:]

  return J_tot















  
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
  if(False):
    log_filename = './logs/rod-dt-%f-N-%d-scheme-%s-g-%s-%s.log' % (dt, n_steps, args.scheme, args.gravity_factor, args.data_name)
    progress_logger = logging.getLogger('Progress Logger')
    progress_logger.setLevel(logging.INFO)
    # Add the log message handler to the logger
    logging.basicConfig(filename=log_filename, level=logging.INFO, filemode='w')
    sl = StreamToLogger(progress_logger, logging.INFO)
    sys.stdout = sl
    sl = StreamToLogger(progress_logger, logging.ERROR)
    sys.stderr = sl

  # Gather parameters to save
  #params = {'A': A, 'ETA': ETA, 'M': M,
            #'REPULSION_STRENGTH_WALL': REPULSION_STRENGTH_WALL,
            #'DEBYE_LENGTH_WALL': DEBYE_LENGTH_WALL, 'dt': dt, 'n_steps': n_steps,
            #'gfactor': args.gravity_factor, 'scheme': args.scheme,
            #'KT': KT}
  params = {'A': A, 'ETA': ETA, 
	  'REPULSION_STRENGTH_WALL': REPULSION_STRENGTH_WALL,
	  'DEBYE_LENGTH_WALL': DEBYE_LENGTH_WALL, \
	  'REPULSION_STRENGTH_BLOBS': REPULSION_STRENGTH_BLOBS,\
	  'DEBYE_LENGTH_BLOBS': DEBYE_LENGTH_BLOBS, \
	  'ATTRACTION_STRENGTH_BLOBS': ATTRACTION_STRENGTH_BLOBS,\
	  'ATTRACTION_LENGTH_BLOBS': ATTRACTION_LENGTH_BLOBS, \
	   'dt': dt, 'n_steps': n_steps, \
	  'gfactor': args.gravity_factor, 'scheme': args.scheme, \
	  'KT': KT}

  print "Parameters for this run are: ", params

  # Script to run the various integrators on the quaternion.
  initial_location = [[-0.048697196167400,   0.081515301071100, 0.350000000000000e0 ], \
                      [0.049069154334600,   0.721085590441000, 0.350000000000000e0]]
  		    
  initial_orientation = [Quaternion([1., 0., 0., 0.]), \
                         Quaternion([1, 0., 0., 0.]),]
  #initial_orientation = [Quaternion([1., 0., 0., 0.]), 
                         #Quaternion([0.707106781186548, 0., 0., 0.707106781186548]),]
  quaternion_integrator = QuaternionIntegrator(rod_mobility,
                                               initial_orientation, 
                                               rod_torque_calculator, 
                                               has_location=True,
                                               initial_location=initial_location,
                                               force_calculator=rod_force_calculator,
                                               slip_velocity=slip_velocity,
                                               resistance_blobs=resistance_blobs,
                                               force_slip=Compute_force_slip)
  quaternion_integrator.kT = KT
  quaternion_integrator.check_function = rod_check_function

  trajectory = [[], []]

  if len(args.data_name) > 0:
    def generate_trajectory_name(scheme):
      trajectory_dat_name = 'rod_trajectory_dt_%g_N_%d_scheme_%s_g_%s_%s.txt' % (dt, n_steps, scheme, args.gravity_factor, args.data_name)
      return trajectory_dat_name
  else:
    def generate_trajectory_name(scheme):
      trajectory_dat_name = 'rod_trajectory_dt_%g_N_%d_scheme_%s-g_%s.txt' % (dt, n_steps, scheme, args.gravity_factor)
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
	veltot = quaternion_integrator.veltot[l]
	omegatot = quaternion_integrator.omegatot[l]
	my_orientation = current_orientation[l].entries
	f.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s \n' % (
	  current_location[l][0], current_location[l][1], current_location[l][2], \
	  my_orientation[0], my_orientation[1], my_orientation[2], my_orientation[3],\
	  veltot[0], veltot[1], veltot[2],\
	  omegatot[0], omegatot[1], omegatot[2]))
       
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






  if args.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

 

