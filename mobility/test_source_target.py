import numpy as np
import sys
import mobility as mob



if __name__ == '__main__':
  print('# Start')
  # Set parameters
  N_src = 10
  N_trg = 12
  N = N_src + N_trg
  eta = 0.13
  a = 0.97

  # Create vectors
  r_src = np.random.rand(N_src, 3)
  r_trg = np.random.rand(N_trg, 3)
  radius_src = np.ones(N_src) * a
  radius_trg = np.ones(N_trg) * a
  force_src = np.random.normal(0, 1, N_src * 3).reshape((N_src, 3))
  # print('force_trg = ', force_trg)

  # Concatenate vectors
  r_vectors = np.concatenate([r_src, r_trg])
  radius = np.concatenate([radius_src, radius_trg])
  force = np.concatenate([force_src, np.zeros((N_trg, 3))])

  # Wall interactions
  if True:
    velocity_py = mob.single_wall_fluid_mobility_product(r_vectors, force.flatten(), eta, a).reshape((N, 3))
    velocity_cpp = mob.single_wall_mobility_trans_times_force_cpp(r_vectors, force, eta, a).reshape((N, 3))
    velocity_numba = mob.single_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a).reshape((N, 3))
    velocity_trg = mob.single_wall_mobility_trans_times_force_source_target_numba(r_src, r_trg, force_src, radius_src, radius_trg, eta).reshape((N_trg, 3))
    velocity_radii = mob.mobility_radii_trans_times_force(r_vectors, force, eta, a, radius, mob.single_wall_mobility_trans_times_force_source_target_numba).reshape((N, 3))

    # Print diff
    print('Near Wall')
    print('|cpp - py|   = ', np.linalg.norm(velocity_cpp - velocity_py))
    print('|numba - py| = ', np.linalg.norm(velocity_numba - velocity_py))
    print('|src - py|   = ', np.linalg.norm(velocity_trg - velocity_py[N_src:]))
    print('|radii - py| = ', np.linalg.norm(velocity_radii - velocity_py))
    print(' ')
    print('velocity_trg = ', velocity_trg[0])
    print(' ')


  # Unbounded
  if True:
    velocity_py = mob.no_wall_fluid_mobility_product(r_vectors, force.flatten(), eta, a).reshape((N, 3))
    velocity_numba = mob.no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a).reshape((N, 3))
    velocity_trg = mob.no_wall_mobility_trans_times_force_source_target_numba(r_src, r_trg, force_src, radius_src, radius_trg, eta).reshape((N_trg, 3))
    velocity_radii = mob.mobility_radii_trans_times_force(r_vectors, force, eta, a, radius, mob.no_wall_mobility_trans_times_force_source_target_numba).reshape((N, 3))

    # Print diff
    print('Unbounded')
    print('|numba - py| = ', np.linalg.norm(velocity_numba - velocity_py))
    print('|src - py|   = ', np.linalg.norm(velocity_trg - velocity_py[N_src:]))
    print('|radii - py| = ', np.linalg.norm(velocity_radii - velocity_py))
    print(' ')
    print('velocity_trg = ', velocity_trg[0])
    print(' ')
    
