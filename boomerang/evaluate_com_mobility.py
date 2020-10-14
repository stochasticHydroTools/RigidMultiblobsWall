''' 
Evaluate the location of the Center of Mobility (CoM). The code finds
the vector r from the location (cross point for the 15-blob boomerang) to the CoM.

The linear system to solve can be found from the Eq. 5.30 in Steven Delong's thesis
M_wF(location 2) = M_wF(location 1) + M_wT(location 1) x r_12(vector from location 1 to location 2)

and the condition (M_wF)^T = M_wF




The result is
r= [  6.43600016e-01   6.43600016e-01   1.40626694e-11]

tip0= [ 2.1 0.   0.]
tip1= [ 0.  2.1  0.]
alpha= 6.43600016e-01
beta = 6.43600016e-01
'''
import numpy as np
import boomerang as bm
from quaternion_integrator.quaternion import Quaternion

if __name__ == '__main__':
  
  # Define Levi-Civita symbol
  eijk = np.zeros( (3, 3, 3) )
  eijk[0,1,2] = 1
  eijk[1,2,0] = 1
  eijk[2,0,1] = 1
  eijk[1,0,2] = -1
  eijk[0,2,1] = -1
  eijk[2,1,0] = -1

  # Define location and orientation
  location = [0., 0., 100000.]
  theta = np.random.normal(0., 1., 4)
  #orientation = Quaternion(theta/np.linalg.norm(theta))
  orientation = Quaternion([1., 0., 0., 0.])
  
  # Get blobs vectors
  r_vectors = bm.get_boomerang_r_vectors_15(location, orientation)

  # Compute mobility at the cross point
  mobility = bm.force_and_torque_boomerang_mobility(r_vectors, location)
           
  # Get sub-matrix M^wF
  mobility_wF = mobility[3:6, 0:3]

  # Get sub-matrix M^wT
  mobility_wT = mobility[3:6, 3:6]

  # Compute matrix A = M^wF - (M^wF).T
  A = mobility_wF - mobility_wF.T

  # Compute matrices B = epsilon_ikl * M^wT_jk - epsilon_jkl * M^wT_ik for l=0,1,2
  B = np.zeros( (3,3,3) )
  for l in range(3):
    for i in range(3):
      for j in range(3):
        for k in range(3):
          B[l,i,j] += eijk[i,k,l] * mobility_wT[j,k] - eijk[j,k,l] * mobility_wT[i,k]


  # Build linear system to find r
  # Build RHS
  RHS = np.zeros(3)
  RHS[0] = A[0,1]
  RHS[1] = A[0,2]
  RHS[2] = A[1,2]
  # Build matrix
  C = np.zeros( (3,3) )
  for i in range(3):
    C[0,i] = B[i,0,1]
    C[1,i] = B[i,0,2]
    C[2,i] = B[i,1,2]
  
  r = np.linalg.solve(C,RHS)
  print('locarion=', location)
  print('CoM respect location r=', r, '\n')

  # Define vectors from cross point to tips
  tip0 = r_vectors[0]  - location
  tip1 = r_vectors[14] - location
  print('boomerang tip0=', tip0)
  print('boomerang tip1=', tip1)
  print('\nif location_CoM = location + alpha * tip0/norm(tip0) + beta * tip1/norm(tip1)')
  print('alpha=', ((r[0]*tip0[0]) + (r[1]*tip0[1]) + (r[2]*tip0[2])) / np.sqrt(tip0[0]**2 + tip0[1]**2 + tip0[2]**2))
  print('beta =', ((r[0]*tip1[0]) + (r[1]*tip1[1]) + (r[2]*tip1[2])) / np.sqrt(tip1[0]**2 + tip1[1]**2 + tip1[2]**2))

  #print mobilities
  print('\n\nMobility defined at the location')
  mobility = bm.force_and_torque_boomerang_mobility(r_vectors, location)
  for i in range(6):
    print(mobility[i, 0], mobility[i, 1], mobility[i, 2], mobility[i, 3], mobility[i, 4], mobility[i, 5])

  print '\n\nMobility defined at the CoM'
  mobility = bm.force_and_torque_boomerang_mobility(r_vectors, location+r)
  for i in range(6):
    print(mobility[i, 0], mobility[i, 1], mobility[i, 2], mobility[i, 3], mobility[i, 4], mobility[i, 5])
  
  print('\nAverage translational mobility at CoM = ', (mobility[0, 0]+mobility[1, 1]+mobility[2, 2])/3.0)
  print('#END')
