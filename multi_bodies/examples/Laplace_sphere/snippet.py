'''
Code snippet to save the concentration field to a vtk file.
'''

    # Try to import the visit_writer (boost implementation)
    try:
      # import visit.visit_writer as visit_writer
      from visit import visit_writer as visit_writer
    except ImportError as e:
      pass  

    if True:
      # Compute concentration on a rectangular write and save to vtk
      # Prepare grid values
      grid = [-5, 5, 100, -5, 5, 100, 0, 0, 1]
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

      # Compute concentration on the grid
      c_grid = background_Laplace[0] + np.einsum('j,ij->i', background_Laplace[1:4], grid_coor) \
        + np.einsum('ik,ik->i', grid_coor, np.einsum('kj,ij->ik', Hessian, grid_coor))
      c_grid -= Laplace_kernels.no_wall_laplace_single_layer_operator_source_target_numba(r_vectors,
                                                                                          grid_coor,
                                                                                          reaction_rate * c / diffusion_coefficient - emitting_rate,
                                                                                          weights)
      c_grid -= Laplace_kernels.no_wall_laplace_double_layer_operator_source_target_numba(r_vectors,
                                                                                          grid_coor,
                                                                                          c,
                                                                                          weights,
                                                                                          normals)
  
      # Prepara data for VTK writer 
      variables = [np.reshape(c_grid, c_grid.size)] 
      dims = np.array([grid_points[0]+1, grid_points[1]+1, grid_points[2]+1], dtype=np.int32) 
      nvars = 1
      vardims = np.array([1])
      centering = np.array([0])
      varnames = ['concentration\0']
      name = output_name + '.concentration_field.vtk'
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
