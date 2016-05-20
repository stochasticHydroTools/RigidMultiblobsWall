''' 
This codes plots the velocity field created by a rod with slip.
To increase the resolution of the visualization
The output is formatted to be visualize with VisIt.
'''

import numpy as np
import sys
import os


import multi_bodies_velocity_field as rod
from quaternion_integrator.quaternion import Quaternion
from mobility import mobility as mb
import visit.visit_writer as visit_writer


# Length from rod to plot velocities
length = [6.1, 6.1, 1.7769]

# Number of grid points along each direction where compute velocity
grid = [np.array([86, 86, 3])]


force_ext = [0.0, 0.0, 0.0]
torque_ext = [0.0, 0.409355878708815, 0.0]



# Define viscosity and rod parameters
#rod.ETA                = 8.9e-4
#rod.A                  = 0.265*np.sqrt(3./2.)
#TOTAL_MASS             = 1.1781*0.0000000002*(9.8*1.e6)
#rod.M                  = [TOTAL_MASS/10. for _ in range(10)]
#rod.REPULSION_STRENGTH = 7.5 * 300.*1.3806488e-5 #rod.KT
#rod.DEBYE_LENGTH       = 0.5 * rod.A


def solve_rigid_body(mobility,
                     force_calculator,
                     torque_calculator,
                     slip_velocity,
                     resistance_blobs,
                     K_matrix,
                     location,
                     orientation):
  '''
  This function solve the rigid body problem for one body and
  it returns the constraint foces (lambda_blobs).
  '''

  # Get mobility matrix of the body
  mobility_body = mobility(location, orientation)

  # Compute external forces
  force = force_calculator(location, orientation)
  force += force_ext
  
  torque = torque_calculator(location, orientation)
  torque += torque_ext
  # Add forces due to slip
  # 0. Get slip on each blob
  slip = slip_velocity(location, orientation)
  #for n in range(len(slip)):
    #if n>35:
     #slip[n] =0.0
  #print "slip = ", slip
  #raw_input()
  # 1. Compute constraint forces due to slip lambda_tile = M^{-1} \cdot slip
  resistance = resistance_blobs(location, orientation)
  lambda_tilde = np.dot(resistance, slip)

  # 2. Compute total effective force due to slip = K^* \cdot lambda_tilde
  K = K_matrix(location, orientation)
  force_slip = -np.dot(K.T, lambda_tilde)
  #print "force_slip = ", force_slip 
  #raw_input()

  # Compute total deterministic force
  force_deterministic = np.concatenate([force, torque]) + force_slip
  print "force_deterministic = ", force_deterministic
  #force_deterministic = force_slip

  # Compute deterministic velocity
  velocity_deterministic = np.dot(mobility_body, force_deterministic)
  print "velocity_deterministic = ", velocity_deterministic
  #raw_input()
  # Compute fluid velocity on the blobs
  velocity_blobs = slip + np.dot(K, velocity_deterministic)

  
  # Compute constraint forces
  lambda_blobs = np.dot(resistance, velocity_blobs)


  return (lambda_blobs,velocity_deterministic)




def fluid_velocity(mobility_blobs, 
                   get_vectors, 
                   get_vectors_shells,
                   lambda_blobs, 
                   location, 
                   orientation, 
                   grid,
                   ETA,
                   A,
                   A_blob_shell,
                   resolution_sphere):
  '''
  This function computes the fluid velocity in the nodes created
  by the constraint foces (lambda_blobs).
  '''
  
  # Create array to store velocities
  vectorVelocities = np.zeros(3*grid[0][0]*grid[0][1]*grid[0][2])
 
  # Compute CoM of the system
  # Get rod coordinates
  r_blobs = []
  CoM = np.zeros(3)

  number_of_rods = len(location)
  for m in range(number_of_rods):
    r_blobs.append(get_vectors(location[m], orientation[m]))
    CoM += sum(r_blobs[m])

  number_of_blobs_per_rod = len(r_blobs[0])
  number_of_blobs = number_of_rods* len(r_blobs[0])
  CoM = CoM/float(number_of_blobs)
  
  if resolution_sphere>0:
    r_blobs_shell = []
    for m in range(number_of_rods):
      for l in range(number_of_blobs_per_rod):
	r_blobs_shell.append(get_vectors_shells(r_blobs[m][l]))
      
  
    number_of_blob_per_shell = len(r_blobs_shell[0])
    lambda_new = np.zeros(3*number_of_blobs*number_of_blob_per_shell+3)
    #print "len(r_blobs_shell) = ", len(r_blobs_shell)
    print "number_of_blob_per_shell = ", number_of_blob_per_shell
    print "len(lambda_new) = ", len(lambda_new)

    for m in range(number_of_blobs):
      for l in range(number_of_blob_per_shell):
	lambda_new[3*m*number_of_blob_per_shell+l*3:3*m*number_of_blob_per_shell+(l+1)*3] = \
	  lambda_blobs[3*m:3*m+3]/float(number_of_blob_per_shell)
	

  # Compute grid coordinates, center box around body
  length_box = length
  meshwidth = length / grid[0]
  length_box_z = length_box[2]
  if(-0.5 * length_box_z + CoM[2] + meshwidth[2] < 0):
    length_box_z = 2.0*CoM[2] - 2.0*A_blob_shell - meshwidth[2] # The box has to be above the wall
  array = [grid[0][0], grid[0][1], grid[0][2], 3]
  grid_coor = np.zeros( array )
  xmesh = np.zeros( grid[0][0]+1)
  ymesh = np.zeros( grid[0][1]+1)
  zmesh = np.zeros( grid[0][2]+1)
  for i in range(grid[0][0]):
    xmesh[i] = -length_box[0] * 0.5 + (i) * meshwidth[0] + CoM[0]
    if (abs(xmesh[i])<1e-8):
       xmesh[i] = 0.0
    for j in range(grid[0][1]):
      ymesh[j] = -length_box[1] * 0.5 + (j) * meshwidth[1] + CoM[1]
      if (abs(ymesh[j])<1e-8):
       ymesh[j] = 0.0
      for k in range(grid[0][2]):
        zmesh[k] = -length_box_z * 0.5 + (k) * meshwidth[2] + CoM[2]
        if (abs(zmesh[k])<1e-8):
	  zmesh[k] = 0.0
        grid_coor[i, j, k, 0] = xmesh[i] + 0.5* meshwidth[0]
        grid_coor[i][j][k][1] = ymesh[j] + 0.5* meshwidth[1]  
        grid_coor[i][j][k][2] = zmesh[k] + 0.5* meshwidth[2]
        #print "k, grid_coor[i][j][k][2] = ", k, grid_coor[i][j][k][2]
        #raw_input()
  
  
  xmesh[grid[0][0]] =  xmesh[grid[0][0]-1] + meshwidth[0]
  ymesh[grid[0][1]] =  ymesh[grid[0][1]-1] + meshwidth[1]
  zmesh[grid[0][2]] =  zmesh[grid[0][2]-1] + meshwidth[2]

  if resolution_sphere>0:
    # Extend rod coordinates with one extra position to measure the fluid velocity at that location
    r_blobs_shell.append([np.array([0., 0., 0.])])

    # Compute the fluid velocity in every grid node
    for i in range(grid[0][0]):
      print "x node ", i, "/", grid[0][0]
      for j in range(grid[0][1]):
	for k in range(grid[0][2]):
	
	  # Select point to compute velocity
	  r_blobs_shell[number_of_blobs][0][0] = grid_coor[i, j, k, 0]
	  r_blobs_shell[number_of_blobs][0][1] = grid_coor[i, j, k, 1]
	  r_blobs_shell[number_of_blobs][0][2] = grid_coor[i, j, k, 2]

	  # Rearrange the vector of blob positions for the mobility computation  
	  r_vec_for_mob = []
	  for l in range(number_of_blobs+1):
	      r_vec_for_mob += r_blobs_shell[l]

	 
	  # Use directly matrix-vector product in C++ instead and compute only the entry 
	  # corresponding to the free blob with index "number_of_blobs*number_of_blob_per_shell"
	  velocity = mb.boosted_mobility_vector_product_one_particle(\
				      r_vec_for_mob, \
				      ETA, \
				      A_blob_shell, \
				      lambda_new, \
				      number_of_blobs*number_of_blob_per_shell)
	  
	  index = grid[0][0]*grid[0][1]*k + grid[0][0]*j + i
	  vectorVelocities[index*3]     = velocity[0]
	  vectorVelocities[index*3 + 1] = velocity[1]
	  vectorVelocities[index*3 + 2] = velocity[2]
  else:
    # Extend rod coordinates with one extra position to measure the fluid velocity at that location
    r_blobs.append([np.array([0., 0., 0.])])

    # Extend blob's forces with a zero component
    lambda_blobs = np.append(lambda_blobs, [0., 0., 0.])

    # Comput the fluid velocity in every grid node
    for i in range(grid[0][0]):
      print "x node ", i, "/", grid[0][0]
      for j in range(grid[0][1]):
	for k in range(grid[0][2]):
	  # Select point to compute velocity
	  r_blobs[number_of_rods][0][0] = grid_coor[i, j, k, 0]
	  r_blobs[number_of_rods][0][1] = grid_coor[i, j, k, 1]
	  r_blobs[number_of_rods][0][2] = grid_coor[i, j, k, 2]
          
	  # Rearrange the vector of blob positions for the mobility computation  
	  r_vec_for_mob = []
	  for l in range(number_of_rods+1):
	      r_vec_for_mob += r_blobs[l]

	  # Use directly matrix-vector product in C++ instead and compute only the entry 
	  # corresponding to the free blob with index "number_of_blobs"
	  velocity = mb.boosted_mobility_vector_product_one_particle(\
				      r_vec_for_mob, \
				      ETA, \
				      A, \
				      lambda_blobs, \
				      number_of_blobs)

	    
	    
	  index = grid[0][0]*grid[0][1]*k + grid[0][0]*j + i
	  vectorVelocities[index*3]     = velocity[0]
	  vectorVelocities[index*3 + 1] = velocity[1]
	  vectorVelocities[index*3 + 2] = velocity[2]
	  #vectorVelocities[index*3]     = velocity[number_of_blobs*3]
	  #vectorVelocities[index*3 + 1] = velocity[number_of_blobs*3 + 1]
	  #vectorVelocities[index*3 + 2] = velocity[number_of_blobs*3 + 2]

  print 'fluid_velocity    -----   DONE'

  return (xmesh,ymesh,zmesh,vectorVelocities)




def saveVTK(location, orientation, slip_velocity):
  '''
  Function to save fluid velocity field in VTK format.
  '''
  


  # Define variables to create VTK file
  vectorVelocities = np.zeros(3*grid[0][0]*grid[0][1]*grid[0][2])
  nameVelocity = "velocity\0";
  nvars = 1
  vardims = np.array([3])
  centering = np.array([0])
  varnames = [nameVelocity]
  dims = np.array([grid[0][0]+1, grid[0][1]+1, grid[0][2]+1]) 
  #dims = np.array([grid[0][0], grid[0][1], grid[0][2]]) 

  meshwidth = length / grid[0]
  print "meshwidth =", meshwidth
  

  numberOfNodes = grid[0][0]*grid[0][1]*grid[0][2]

  # Create ad hoc velocity for testing
  if(0):
    for i in range(numberOfNodes):
      

      # vectorVelocities[3*i:3*i+3] = np.random.normal(0., 1., 3) # X, Y and Z
      kx = i % grid[0][0]
      ky = (i % (grid[0][0]*grid[0][1])) / grid[0][0]
      kz = i / (grid[0][0] * grid[0][1]);
      vectorVelocities[3*i]   = kx
      vectorVelocities[3*i+1] = ky
      vectorVelocities[3*i+2] = kz
      
  print 'compute lambdas'   
  # Solve rigid body problem and return constraint forces
  (lambda_blobs,vel_body) = solve_rigid_body(rod.sphere_mobility,
		                  #rod.rod_mobility,
                                  rod.sphere_force_calculator,
                                  rod.sphere_torque_calculator,
                                  #rod.rod_force_calculator,
                                  #rod.rod_torque_calculator,
                                  rod.slip_velocity_sphere,
                                  #rod.slip_velocity,
                                  #rod.resistance_blobs,
                                  rod.resistance_blobs_sphere,
                                  rod.K_matrix_sphere,
                                  #rod.K_matrix,
                                  location,
                                  orientation) 
  print 'compute lambdas    -----   DONE'   
  print 'compute blob velocities' 
  # Compute fluid velocity on the grid using the constraint forces 
  (x,y,z,vectorVelocities) = fluid_velocity(mb.boosted_single_wall_fluid_mobility, 
                                    rod.get_sphere_r_vectors,
                                    rod.get_sphere_for_sphere_r_vectors,
                                    #rod.get_rod_r_vectors,
                                    #rod.get_sphere_for_rod_r_vectors,
                                    lambda_blobs, 
                                    location, 
                                    orientation, 
                                    grid,
                                    rod.ETA,
                                    rod.A,
                                    rod.A_blob_shell,
                                    rod.resolution_sphere)
  
  xmesh = np.array(x)
  ymesh = np.array(y)
  zmesh = np.array(z)
  
  

  variables = [vectorVelocities]
  
  # Open file
  number_of_rods = len(location)
  r_blobs = []
  CoM = np.zeros(3)
  for m in range( number_of_rods):
    #r_blobs.append(rod.get_rod_r_vectors(location[m], orientation[m]))
    r_blobs.append(rod.get_sphere_r_vectors(location[m], orientation[m]))

    CoM += sum(r_blobs[m])

  number_of_blobs_per_rod = len(r_blobs[0])
  number_of_blobs = len(r_blobs[0]) *  number_of_rods
  CoM = CoM/float(number_of_blobs)
  
  slip = slip_velocity(location, orientation)
  Nslip = 0
  for l in range(number_of_rods):
    for i in range(number_of_blobs_per_rod):
      slip_blob = slip[l*3*number_of_blobs_per_rod + 3*i:l*3*number_of_blobs_per_rod+3*(i+1)]
      sum_slip_blob = slip_blob[0] + slip_blob[1] + slip_blob[2]
      if abs(sum_slip_blob)>1e-5:
	Nslip += 1
	

  
  name_dir = 'Results_flow_fields/Nrods_' + str(number_of_rods) \
          + '_Nblobs_' + str(number_of_blobs_per_rod) \
	  + '_dz_' + str(CoM[2]) \
	  + '_box_size_' + str(length[0]) + '_'  + str(length[1]) + '_' + str(length[2])  \
	  + '_grid_'  + str(grid[0][0]) + '_'  + str(grid[0][1]) + '_' + str(grid[0][2])  \
	  + '_rep_wall_' + str(rod.REPULSION_STRENGTH_WALL) \
	  + '_Nslip_'  + str(Nslip) \
	  + '_res_shell_' + str(rod.resolution_sphere) \
	  + '_force_ext_' + str(force_ext[0]) + '_'  + str(force_ext[1]) + '_' + str(force_ext[2])
	
  if not os.path.exists(name_dir):
    os.makedirs(name_dir)
    print 'Created directory : ', name_dir

  
  name = 'velocity_Nrods_' + str(number_of_rods) \
          + '_Nblobs_' + str(number_of_blobs_per_rod) \
	  + '_dz_' + str(CoM[2]) \
	  + '_box_size_' + str(length[0]) + '_'  + str(length[1]) + '_' + str(length[2])  \
	  + '_grid_'  + str(grid[0][0]) + '_'  + str(grid[0][1]) + '_' + str(grid[0][2])  \
	  + '_rep_wall_' + str(rod.REPULSION_STRENGTH_WALL) \
	  + '_Nslip_'  + str(Nslip) \
	  + '_res_shell_' + str(rod.resolution_sphere) \
	  + '_force_ext_' + str(force_ext[0]) + '_'  + str(force_ext[1]) + '_' + str(force_ext[2]) \
	  + '.vtk'
  name = name_dir + '/' + name
  print 'name = ', name
  


  # Write velocity field
  visit_writer.boost_write_rectilinear_mesh(name,      # File's name
                                        0,         # 0=ASCII,  1=Binary
                                        dims,      # {mx, my, mz}
                                        xmesh,         # xmesh
                                        ymesh,         # ymesh
                                        zmesh,         # zmesh
                                        nvars,     # Number of variables
                                        vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                        centering, # Write to cell centers of corners
                                        varnames,  # Variables' names
                                        variables) # Variables
  
  
  name_txt = 'velocity_Nrods_' + str(number_of_rods) \
          + '_Nblobs_' + str(number_of_blobs_per_rod) \
	  + '_dz_' + str(CoM[2]) \
	  + '_box_size_' + str(length[0]) + '_'  + str(length[1]) + '_' + str(length[2])  \
	  + '_grid_'  + str(grid[0][0]) + '_'  + str(grid[0][1]) + '_' + str(grid[0][2])  \
	  + '_rep_wall_' + str(rod.REPULSION_STRENGTH_WALL) \
	  + '_Nslip_'  + str(Nslip) \
	  + '_res_shell_' + str(rod.resolution_sphere) \
	  + '_force_ext_' + str(force_ext[0]) + '_'  + str(force_ext[1]) + '_' + str(force_ext[2]) \
	  + '.txt'
  name_txt = name_dir + '/' + name_txt
  print 'name_txt = ', name_txt
  f = open(name_txt, 'w')
  #data  = str(dims[0]-1) + ' '\
        #+ str(dims[1]-1) + ' '\
	#+ str(dims[2]-1) + ' '\
	#+ str(vel_body[0]) + ' '\
	#+ str(vel_body[1]) + ' '\
	#+ str(vel_body[2]) + '\n'  
      
  data  = str(dims[0]-1) + ' '\
        + str(dims[1]-1) + ' '\
	+ str(dims[2]-1) + ' '
  for i in range(len(vel_body)):
    data = data + str(vel_body[i])+ ' '
  data = data +'\n'
	
  for i in range(dims[0]-1):
    for j in range(dims[1]-1):
      for k in range(dims[2]-1):
	index =(dims[0]-1)*(dims[1]-1)*k + (dims[0]-1)*j + i
	data = data + str(xmesh[i]+meshwidth[0]/2.0) + ' ' \
	            + str(ymesh[j]+meshwidth[1]/2.0) + ' ' \
		    + str(zmesh[k]+meshwidth[2]/2.0) + ' ' \
		    + str(vectorVelocities[3*index])+ ' ' \
		    + str(vectorVelocities[3*index+1])+ ' ' \
		    + str(vectorVelocities[3*index+2]) + '\n'
		  
  f.write(data)
  # Close file
  f.close()
  
  name_vel_body = name_dir + '/' +'vel_body.txt'
  print 'name_vel_body = ', name_vel_body
  f = open(name_vel_body, 'w')
  data  = str(vel_body[0]) + ' '\
	+ str(vel_body[1]) + ' '\
	+ str(vel_body[2]) + '\n'
      
  f.write(data)
  # Close file
  f.close()

def saveBlobsXYZ(location, orientation, slip_velocity):
  '''
  Function to save the blobs' coordinates in XYZ format to
  be visualized with visit. The format is:
  number_of_blobs
  one_line_of_text
  type_blob_0 x_0 y_0 z_0
  type_blob_1 x_1 y_1 z_1
  .
  .
  .
  '''
  
  slip = slip_velocity(location, orientation)


  number_of_rods = len(location)
  

  r_blobs = []
  CoM = np.zeros(3)
  for m in range( number_of_rods):
    r_blobs.append(rod.get_sphere_r_vectors(location[m], orientation[m]))
    #r_blobs.append(rod.get_rod_r_vectors(location[m], orientation[m]))

    CoM += sum(r_blobs[m])

  number_of_blobs_per_rod = len(r_blobs[0])
  number_of_blobs = len(r_blobs[0]) *  number_of_rods
  CoM = CoM/float(number_of_blobs)
  
  Nslip = 0
  for l in range(number_of_rods):
    for i in range(number_of_blobs_per_rod):
      slip_blob = slip[l*3*number_of_blobs_per_rod + 3*i:l*3*number_of_blobs_per_rod+3*(i+1)]
      sum_slip_blob = slip_blob[0] + slip_blob[1] + slip_blob[2]
      if abs(sum_slip_blob)>1e-5:
	Nslip += 1

  name_dir = 'Results_flow_fields/Nrods_' + str(number_of_rods) \
          + '_Nblobs_' + str(number_of_blobs_per_rod) \
	  + '_dz_' + str(CoM[2]) \
	  + '_box_size_' + str(length[0]) + '_'  + str(length[1]) + '_' + str(length[2])  \
	  + '_grid_'  + str(grid[0][0]) + '_'  + str(grid[0][1]) + '_' + str(grid[0][2])  \
	  + '_rep_wall_' + str(rod.REPULSION_STRENGTH_WALL) \
	  + '_Nslip_'  + str(Nslip) \
	  + '_res_shell_' + str(rod.resolution_sphere) \
	  + '_force_ext_' + str(force_ext[0]) + '_'  + str(force_ext[1]) + '_' + str(force_ext[2])
	
  if not os.path.exists(name_dir):
    os.makedirs(name_dir)
    
  filename = 'blobs_Nrods_' + str(number_of_rods) \
          + '_Nblobs_' + str(number_of_blobs_per_rod) \
	  + '_dz_' + str(CoM[2]) \
	  + '_box_size_' + str(length[0]) + '_'  + str(length[1]) + '_' + str(length[2]) \
	  + '_grid_'  + str(grid[0][0]) + '_'  + str(grid[0][1]) + '_' + str(grid[0][2]) \
	  + '_Nslip_'  + str(Nslip) \
	  + '_res_shell_' + str(rod.resolution_sphere) \
	  + '.xyz'  
  filename = name_dir + '/' + filename
  print 'filename = ', filename

  f = open(filename, 'w')
  data  = str(number_of_blobs) + '\n'
  data += 'any line of text \n'
  for l in range(number_of_rods):
    for i in range(number_of_blobs_per_rod):
      slip_blob = slip[l*3*number_of_blobs_per_rod + 3*i:l*3*number_of_blobs_per_rod+3*(i+1)]
      sum_slip_blob = slip_blob[0] + slip_blob[1] + slip_blob[2]
      if sum_slip_blob>1e-5:
	element = 'O '
      elif sum_slip_blob<-1e-5:
	element = 'N '
      else:
	element = 'C '
      data += element \
	   + str(r_blobs[l][i][0]) + ' ' \
	   + str(r_blobs[l][i][1]) + ' ' \
	   + str(r_blobs[l][i][2]) + ' ' \
	   + str(rod.A) + '\n' 
  f.write(data)
  f.close()


  filename_txt = 'blobs_Nrods_' + str(number_of_rods) \
          + '_Nblobs_' + str(number_of_blobs_per_rod) \
	  + '_dz_' + str(CoM[2]) \
	  + '_box_size_' + str(length[0]) + '_'  + str(length[1]) + '_' + str(length[2]) \
	  + '_grid_'  + str(grid[0][0]) + '_'  + str(grid[0][1]) + '_' + str(grid[0][2]) \
	  + '_Nslip_'  + str(Nslip) \
	  + '_res_shell_' + str(rod.resolution_sphere) \
	  + '.txt'  
  filename_txt = name_dir + '/' + filename_txt
  print 'filename_txt = ', filename_txt

  f = open(filename_txt, 'w')
  data  = ''
  for l in range(number_of_rods):
    for i in range(number_of_blobs_per_rod):
      slip_blob = slip[l*3*number_of_blobs_per_rod + 3*i:l*3*number_of_blobs_per_rod+3*(i+1)]
      data += str(r_blobs[l][i][0]) + ' ' \
	   + str(r_blobs[l][i][1]) + ' ' \
	   + str(r_blobs[l][i][2]) + ' ' \
	   + str(slip_blob[0]) + ' ' \
	   + str(slip_blob[1]) + ' ' \
	   + str(slip_blob[2]) + ' ' \
	   + str(rod.A) + '\n' 
  f.write(data)
  f.close()
 
                                          
if __name__ == '__main__':
  
    ## Define rod's location and orientation
  #location = [[0.0,   0.0, 0.312030637748 ],\
             #[0.08072053278559,   0.447748294579267, 0.312030637748  ] ]
  #location = [[0.0,   0.0, 0.2596 ],]
  location = [[0.0,   0.0, 0.711 ],]
             #[0.733333,   0.5, 0.3246] ]

	   
  orientation = [Quaternion([1., 0., 0., 0.]),]
                         #Quaternion([1, 0., 0., 0.])]
  
  saveVTK(location, orientation, rod.slip_velocity_sphere)
  saveBlobsXYZ(location, orientation, rod.slip_velocity_sphere)
  #saveVTK(location, orientation, rod.slip_velocity)
  #saveBlobsXYZ(location, orientation, rod.slip_velocity)
       
  print '#END'
    


