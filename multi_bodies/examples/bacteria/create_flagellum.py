'''
Utility code to create the vertex file of a flagellum as in J. J. Higdon, J. Fluid Mech., 94 (2), 331 (1979).

The centerline of the flagellum has coordinates parametrize by z

x = alpha * (1 - exp(-k_E**2 * z**2) * cos(k*z)
y = alpha * (1 - exp(-k_E**2 * z**2) * sin(k*z)
z = z
'''
import numpy as np
import scipy.integrate as scin
import sys

if __name__ == '__main__':
  # Set parameters
  L = 20.0
  alpha = 0.4
  k = 1.0 / alpha
  k_E = k
  lambda_wave = 2 * np.pi / k
  if False:
    aspect_ratio = 0.03
    # Compute other parameters
    blob_radius = L * aspect_ratio 
  else:
    blob_radius = 0.1310
    # Compute other parameters
    aspect_ratio = blob_radius / L
  N = int(L / (2 * blob_radius))
   
  # Arclength
  def dL(z):
    return np.sqrt(1 + alpha**2 * 4 * k_E**2 * z**2 * np.exp(-2*k_E**2 * z**2) + alpha**2 * (1 - np.exp(-k_E**2 * z**2)**2) * k**2)
  z = np.linspace(0, L, num=int(500 / aspect_ratio) + 1)
  dz = L / z.size
  L_z = scin.cumtrapz(dL(z), z, initial=0)
  
  # Create flagellum with equally spaced blobs
  r_vectors = []
  for i in range(N):
    indx = np.abs(L_z - 2 * i * blob_radius).argmin()
    zi = dz * indx
    r_vectors.append([alpha * (1 - np.exp(-k_E**2 * zi**2))*np.cos(k*zi), alpha * (1 - np.exp(-k_E**2 * zi**2))*np.sin(k*zi), zi])
  r_vectors = np.array(r_vectors)
  # r_vectors[:,2] += 1 + blob_radius

  # Print result
  print('# L            = ', L)
  print('# alpha        = ', alpha)
  print('# k_E          = ', k_E)
  print('# k            = ', k)
  print('# lambda       = ', lambda_wave)
  print('# N_lambda     = ', zi / lambda_wave)
  print('# aspect_ratio = ', aspect_ratio)
  print('# blob_radius  = ', blob_radius)
  print(r_vectors.size // 3)
  np.savetxt(sys.stdout, r_vectors)
    
  
