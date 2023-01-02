
import numpy as np
from mobility import mobility as mob
import general_application_utils as utils
try:
  from pyevtk.hl import gridToVTK
except ImportError:
  pass

# Try to import the visit_writer (boost implementation)
try:
  # import visit.visit_writer as visit_writer
  from visit import visit_writer as visit_writer
except ImportError as e:
  print(e)
  pass

def plot_velocity_field(grid, r_vectors_blobs, lambda_blobs, blob_radius, eta, output, tracer_radius, radius_source = None, frame_body = None, *args, **kwargs):
  '''
  This function plots the velocity field to a grid using boost visit writer
  '''
  # Prepare grid values
  grid = np.reshape(grid, (3,3)).T
  grid_length = grid[1] - grid[0]
  grid_points = np.array(grid[2], dtype=np.int32)
  num_points = grid_points[0] * grid_points[1] * grid_points[2]

  # Set grid coordinates
  dx_grid = grid_length / grid_points
  grid_x = np.array([grid[0,0] + dx_grid[0] * (x+0.5) for x in range(grid_points[0])])
  grid_y = np.array([grid[0,1] + dx_grid[1] * (x+0.5) for x in range(grid_points[1])])
  grid_z = np.array([grid[0,2] + dx_grid[2] * (x+0.5) for x in range(grid_points[2])])
  # Be aware, x is the fast axis.
  zz, yy, xx = np.meshgrid(grid_z, grid_y, grid_x, indexing = 'ij')
  grid_coor = np.zeros((num_points, 3))
  grid_coor[:,0] = np.reshape(xx, xx.size)
  grid_coor[:,1] = np.reshape(yy, yy.size)
  grid_coor[:,2] = np.reshape(zz, zz.size)

  # Transform grid to the body frame of reference
  if frame_body is not None:
    grid_coor = utils.get_vectors_frame_body(grid_coor, body=frame_body)

  # Set radius of blobs (= a) and grid nodes (= 0)
  if radius_source is None:
    radius_source = np.ones(r_vectors_blobs.size // 3) * blob_radius
  radius_target = np.ones(grid_coor.size // 3) * tracer_radius

  # Compute velocity field 
  mobility_vector_prod_implementation = kwargs.get('mobility_vector_prod_implementation')
  if mobility_vector_prod_implementation == 'python':
    grid_velocity = mob.mobility_vector_product_source_target_one_wall(r_vectors_blobs, 
                                                                       grid_coor, 
                                                                       lambda_blobs, 
                                                                       radius_source, 
                                                                       radius_target, 
                                                                       eta, 
                                                                       *args, 
                                                                       **kwargs) 
  elif mobility_vector_prod_implementation == 'C++':
    grid_velocity = mob.boosted_mobility_vector_product_source_target(r_vectors_blobs, 
                                                                      grid_coor, 
                                                                      lambda_blobs, 
                                                                      radius_source, 
                                                                      radius_target, 
                                                                      eta, 
                                                                      *args, 
                                                                      **kwargs)
  elif mobility_vector_prod_implementation == 'numba_no_wall':
    grid_velocity = mob.no_wall_mobility_trans_times_force_source_target_numba(r_vectors_blobs, 
                                                                               grid_coor, 
                                                                               lambda_blobs, 
                                                                               radius_source, 
                                                                               radius_target, 
                                                                               eta, 
                                                                               *args, 
                                                                               **kwargs) 
  else:
    grid_velocity = mob.single_wall_mobility_trans_times_force_source_target_pycuda(r_vectors_blobs, 
                                                                                    grid_coor, 
                                                                                    lambda_blobs, 
                                                                                    radius_source, 
                                                                                    radius_target, 
                                                                                    eta, 
                                                                                    *args, 
                                                                                    **kwargs)

  # Tranform velocity to the body frame of reference
  if frame_body is not None:
    grid_velocity = utils.get_vectors_frame_body(grid_velocity, body=frame_body, translate=False, transpose=True)
  
  # Prepara data for VTK writer 
  variables = [np.reshape(grid_velocity, grid_velocity.size)] 
  dims = np.array([grid_points[0]+1, grid_points[1]+1, grid_points[2]+1], dtype=np.int32) 
  nvars = 1
  vardims = np.array([3])
  centering = np.array([0])
  varnames = ['velocity\0']
  name = output + '.velocity_field.vtk'
  grid_x = grid_x - dx_grid[0] * 0.5
  grid_y = grid_y - dx_grid[1] * 0.5
  grid_z = grid_z - dx_grid[2] * 0.5
  grid_x = np.concatenate([grid_x, [grid[1,0]]])
  grid_y = np.concatenate([grid_y, [grid[1,1]]])
  grid_z = np.concatenate([grid_z, [grid[1,2]]])

  

  # Write velocity field
  visit_writer.boost_write_rectilinear_mesh(name,      # File's name
                                            0,         # 0=ASCII,  1=Binary
                                            dims,      # {mx, my, mz}
                                            grid_x,     # xmesh
                                            grid_y,     # ymesh
                                            grid_z,     # zmesh
                                            nvars,     # Number of variables
                                            vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                            centering, # Write to cell centers of corners
                                            varnames,  # Variables' names
                                            variables) # Variables
  return

def plot_velocity_field_pyVTK(grid, r_vectors_blobs, lambda_blobs, blob_radius, eta, output, tracer_radius, *args, **kwargs):
  '''
  This function plots the velocity field to a grid using pyevtk 
  '''
  # Prepare grid values
  grid = np.reshape(grid, (3,3)).T
  grid_length = grid[1] - grid[0]
  grid_points = np.array(grid[2], dtype=np.int32)
  num_points = grid_points[0] * grid_points[1] * grid_points[2]

  # Set grid coordinates
  dx_grid = grid_length / grid_points
  grid_x = np.array([grid[0,0] + dx_grid[0] * (x+0.5) for x in range(grid_points[0])])
  grid_y = np.array([grid[0,1] + dx_grid[1] * (x+0.5) for x in range(grid_points[1])])
  grid_z = np.array([grid[0,2] + dx_grid[2] * (x+0.5) for x in range(grid_points[2])])
  
  xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z, indexing = 'ij')

  grid_coor = np.zeros((num_points, 3))
  grid_coor[:,0] = np.reshape(xx, xx.size)
  grid_coor[:,1] = np.reshape(yy, yy.size)
  grid_coor[:,2] = np.reshape(zz, zz.size)

  # Set radius of blobs (= a) and grid nodes (= 0)
  radius_source = np.ones(r_vectors_blobs.size // 3) * blob_radius 
  radius_target = np.ones(grid_coor.size // 3) * tracer_radius

  # Compute velocity field 
  mobility_vector_prod_implementation = kwargs.get('mobility_vector_prod_implementation')
  print('mobility_vector_prod_implementation = ', mobility_vector_prod_implementation)
  if mobility_vector_prod_implementation == 'python':
    grid_velocity = mob.mobility_vector_product_source_target_one_wall(r_vectors_blobs, 
                                                                       grid_coor, 
                                                                       lambda_blobs, 
                                                                       radius_source, 
                                                                       radius_target, 
                                                                       eta, 
                                                                       *args, 
                                                                       **kwargs) 
  elif mobility_vector_prod_implementation == 'C++':
    grid_velocity = mob.boosted_mobility_vector_product_source_target(r_vectors_blobs, 
                                                                      grid_coor, 
                                                                      lambda_blobs, 
                                                                      radius_source, 
                                                                      radius_target, 
                                                                      eta, 
                                                                      *args, 
                                                                      **kwargs)
  elif mobility_vector_prod_implementation == 'numba_no_wall':
    grid_velocity = mob.no_wall_mobility_trans_times_force_source_target_numba(r_vectors_blobs, 
                                                                               grid_coor, 
                                                                               lambda_blobs, 
                                                                               radius_source, 
                                                                               radius_target, 
                                                                               eta, 
                                                                               *args, 
                                                                               **kwargs) 
  else:
    grid_velocity = mob.single_wall_mobility_trans_times_force_source_target_pycuda(r_vectors_blobs, 
                                                                                    grid_coor, 
                                                                                    lambda_blobs, 
                                                                                    radius_source, 
                                                                                    radius_target, 
                                                                                    eta, 
                                                                                    *args, 
                                                                                    **kwargs) 
  # Prepara data for VTK writer 
  name = output + '.vtk'
  print(name)
  
  grid_velocity  = np.reshape(grid_velocity,(num_points,3)) 
  velx = np.ascontiguousarray(np.reshape(grid_velocity[:,0],(grid_points[0], grid_points[1], grid_points[2])))
  vely = np.ascontiguousarray(np.reshape(grid_velocity[:,1],(grid_points[0], grid_points[1], grid_points[2])))
  velz = np.ascontiguousarray(np.reshape(grid_velocity[:,2],(grid_points[0], grid_points[1], grid_points[2])))

  grid_x = grid_x - dx_grid[0] * 0.5
  grid_y = grid_y - dx_grid[1] * 0.5
  grid_z = grid_z - dx_grid[2] * 0.5
  grid_x = np.ascontiguousarray(np.concatenate([grid_x, [grid[1,0]]]))
  grid_y = np.ascontiguousarray(np.concatenate([grid_y, [grid[1,1]]]))
  grid_z = np.ascontiguousarray(np.concatenate([grid_z, [grid[1,2]]]))

  gridToVTK(name, 
            grid_x, 
            grid_y, 
            grid_z, 
            cellData = {"velocity" : (velx, vely, velz)})

  return
