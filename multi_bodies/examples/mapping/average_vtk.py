'''
Compute the average of vtk files. Before running set the variables at the top of the file.
'''

import numpy as np
import meshio
import sys

# Try to import the visit_writer (boost implementation)
try:
  sys.path.append('../../../')
  from visit import visit_writer as visit_writer
except ImportError as e:
  print(e)
  pass


# Set variables
prefix = './data/run_flagellated.step.'
suffix = '.velocity_field.vtk'
variable_name = 'velocity'
grid = np.array([-6, 6, 50,  -6, 6, 50,  -8, 16, 100])
upper_bound = 1.0e+24
lower_bound = -1.0e+24
out_of_bound_value = 1.0e+20
variable_dimension = 3
start = 0
num_steps = 24
step_size = 1
file_name = prefix + str(start).zfill(8) + suffix

# Create variable for averaging
mesh = meshio.read(file_name)
points = mesh.points
cells = mesh.cells
vel = mesh.cell_data
x = vel[variable_name][0].flatten()
x_avg = np.copy(x)
x_count = np.zeros(x.shape[0])
sel = np.logical_and(x > lower_bound, x < upper_bound)
x_count[sel] += 1

# Loop over files
for i in range(start+step_size, num_steps, step_size):
  name = prefix + str(i).zfill(8)  + suffix
  print(name)
  mesh = meshio.read(name)
  xi = mesh.cell_data[variable_name][0].flatten()
  sel = np.logical_and(xi > lower_bound, xi < upper_bound)
  x_count[sel] += 1
  x_avg[sel] += xi[sel]

# Normalize
sel = x_count > 0
x_avg[sel] = x_avg[sel] / x_count[sel]
x_avg[~sel] = out_of_bound_value

if True:
  # Prepare grid values
  grid = np.reshape(grid[0:9], (3,3)).T
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

  # Prepara data for VTK writer
  variables = [np.reshape(x_avg, x_avg.size)] 
  dims = np.array([grid_points[0]+1, grid_points[1]+1, grid_points[2]+1], dtype=np.int32) 
  nvars = 1
  # vardims = np.array([ 1 if len(x.shape) == 1 else x.shape[1] ])
  vardims = np.array([variable_dimension])
  centering = np.array([0])
  varnames = [variable_name + '\0']
  name = prefix + 'average' + suffix
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
                                            grid_x,    # xmesh
                                            grid_y,    # ymesh
                                            grid_z,    # zmesh
                                            nvars,     # Number of variables
                                            vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                            centering, # Write to cell centers of corners
                                            varnames,  # Variables' names
                                            variables) # Variables
  


