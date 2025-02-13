import numpy as np
import sys
sys.path.append('../')
import mobility as mob


if __name__ == '__main__':
  print('# Start')

  # Set parameters
  N_src = 1
  N_trg = 1
  N = N_src + N_trg
  eta = 1
  a = 1
  z_min = 2
  dx = 1e-06
  wall = 0

  # Create vectors
  r_src = np.random.rand(N_src, 3)  
  r_trg = np.random.rand(N_trg, 3)
  #r_src = np.array([0, 0, 1]).reshape((N_src, 3))
  #r_trg = np.array([0, 0, 2]).reshape((N_trg, 3))
  r_src[:,2] += z_min
  r_trg[:,2] += z_min
  radius_source = np.zeros(N_src)
  radius_target = np.zeros(N_trg)  
  weights = np.ones(N_src) * a  
  vector_src = np.random.normal(0, 1, N_src * 3).reshape((N_src, 3))
  # vector_src[:,:] = 0
  # vector_src[:,2] = 1  
  normals_src = np.random.normal(0, 1, N_src * 3).reshape((N_src, 3))
  # normals_src[:,:] = 0
  # normals_src[:,0] = 1  
  normals_src /= np.linalg.norm(normals_src, axis=1)[:,None]
  print('r_src       = ', r_src)
  print('r_trg       = ', r_trg)  
  print('normals_src = ', normals_src)
  print('vector_src  = ', vector_src)
  print(' ')


  if wall:
    # Test symmetry
    velocity_numba = mob.double_layer_source_target_numba(r_src, r_trg, normals_src, vector_src, weights, wall=wall)
    velocity_finite_np = -mob.single_wall_mobility_trans_times_force_source_target_numba(r_src + normals_src * 0.5 * dx, r_trg, vector_src, radius_source, radius_target, eta) 
    velocity_finite_nm = -mob.single_wall_mobility_trans_times_force_source_target_numba(r_src - normals_src * 0.5 * dx, r_trg, vector_src, radius_source, radius_target, eta)
    velocity_finite_vp = -mob.single_wall_mobility_trans_times_force_source_target_numba(r_src + vector_src * 0.5 * dx, r_trg, normals_src, radius_source, radius_target, eta) 
    velocity_finite_vm = -mob.single_wall_mobility_trans_times_force_source_target_numba(r_src - vector_src * 0.5 * dx, r_trg, normals_src, radius_source, radius_target, eta)
    vn = -np.einsum('bi,bi->b', vector_src, normals_src) / eta 
    velocity_pressure = np.zeros((N_src, 3))
    x = np.zeros((N_src, 3))
    x[:,0] = 1
    velocity_pressure[:,0] = mob.single_wall_pressure_Stokeslet_numba(r_src, r_trg, x) * vn
    x[:,:] = 0
    x[:,1] = 1
    velocity_pressure[:,1] = mob.single_wall_pressure_Stokeslet_numba(r_src, r_trg, x) * vn
    x[:,:] = 0    
    x[:,2] = 1
    velocity_pressure[:,2] = mob.single_wall_pressure_Stokeslet_numba(r_src, r_trg, x) * vn 
    velocity_finite = (velocity_finite_np - velocity_finite_nm + velocity_finite_vp - velocity_finite_vm) / dx + velocity_pressure    
    
    vel_diff = velocity_finite - velocity_numba
    print('Wall test')
  
  else:
    # Test symmetry
    velocity_numba = mob.double_layer_source_target_numba(r_src, r_trg, normals_src, vector_src, weights, wall=wall)
    velocity_finite_np = -mob.no_wall_mobility_trans_times_force_source_target_numba(r_src + normals_src * 0.5 * dx, r_trg, vector_src, radius_source, radius_target, eta) 
    velocity_finite_nm = -mob.no_wall_mobility_trans_times_force_source_target_numba(r_src - normals_src * 0.5 * dx, r_trg, vector_src, radius_source, radius_target, eta)
    velocity_finite_vp = -mob.no_wall_mobility_trans_times_force_source_target_numba(r_src + vector_src * 0.5 * dx, r_trg, normals_src, radius_source, radius_target, eta) 
    velocity_finite_vm = -mob.no_wall_mobility_trans_times_force_source_target_numba(r_src - vector_src * 0.5 * dx, r_trg, normals_src, radius_source, radius_target, eta)
    vn = -np.einsum('bi,bi->b', vector_src, normals_src) / eta 
    velocity_pressure = np.zeros((N_src, 3))
    x = np.zeros((N_src, 3))
    x[:,0] = 1
    velocity_pressure[:,0] = mob.no_wall_pressure_Stokeslet_numba(r_src, r_trg, x) * vn
    x[:,:] = 0
    x[:,1] = 1
    velocity_pressure[:,1] = mob.no_wall_pressure_Stokeslet_numba(r_src, r_trg, x) * vn
    x[:,:] = 0    
    x[:,2] = 1
    velocity_pressure[:,2] = mob.no_wall_pressure_Stokeslet_numba(r_src, r_trg, x) * vn 
    velocity_finite = (velocity_finite_np - velocity_finite_nm + velocity_finite_vp - velocity_finite_vm) / dx + velocity_pressure    
    
    vel_diff = velocity_finite - velocity_numba
    print('No wall test')


  print('vel_numba      = ', velocity_numba)
  print('vel_finite      = ', velocity_finite)
  print(' ')
  print(' ')
  print('vel_finite_n    = ', (velocity_finite_np - velocity_finite_nm) / dx)
  print('vel_finite_v    = ', (velocity_finite_vp - velocity_finite_vm) / dx)    
  print('vel_pressure    = ', velocity_pressure)    
  print(' ')
  print('|vel_diff|     = ', np.linalg.norm(vel_diff))
  print('|vel_diff|_rel =', np.linalg.norm(vel_diff) / np.linalg.norm(velocity_numba))
    
    
