'''
This code allows to create a randomly perturbe
square lattice in qausi 2D.

HOW TO USE:
python create_random_suspension.py Nx Ny Lx Ly z rand_factor

with
Nx = number of particles along x-axis.
Ny = number of particles along y-axis.
Lx = length along x-axis.
Ly = length along y-axis.
z = average height of particles.
rand_factor = amplitude perturbation from the square lattice.
'''

import sys
import numpy as np

if __name__ == '__main__':
  Nx = int(sys.argv[1])
  Ny = int(sys.argv[2])
  Lx = float(sys.argv[3])
  Ly = float(sys.argv[4])
  z_0 = float(sys.argv[5])
  rand_factor = float(sys.argv[6])
  
  dx = Lx / Nx
  dy = Ly / Ny
  dz = z_0

  print(Nx * Ny)
  for k in range(Ny):
    for l in range(Nx):
      x = float(l) * dx + (2.0 * np.random.random() - 1.0) * rand_factor
      y = float(k) * dy + (2.0 * np.random.random() - 1.0) * rand_factor
      z = dz + (2.0 * np.random.random() - 1.0) * rand_factor
      print(x, y, z, 1.0, 0.0, 0.0, 0.0)

