'''
Create articulated input files from vertex file.

Use:

python create_soft_articulated.py file.vertex d_max d_frac

with:
file.vertex =  vertex file to conect with links
d_max = maximum distance to connect blobs with links
d_frac = (between 0 and 1) where to put the joint in the links between two blobs. 0.5 is the middle.


It generates the following files:
create_soft_articulated_histogram.dat = histogram of the blob-blob distances. Usefult to determine right value of d_max.
create_soft_articulated.clones = clones file with the position (and orientation) of the blobs.
create_soft_articulated.const = file with the constrainst information.
create_soft_articulated.list_vertex = file with the list of vertex files.
'''

import numpy as np
import sys
sys.path.append('../')
from body import body
from quaternion_integrator.quaternion import Quaternion


if __name__ == '__main__':
  # Inputs
  r = np.loadtxt(sys.argv[1], skiprows=1)
  d_max = float(sys.argv[2])
  d_frac = float(sys.argv[3])
  if len(sys.argv) > 4:
    method = sys.argv[4]
    max_num_neighbors = int(sys.argv[5])
  else:
    method = 'distance'

  if False:
    theta = np.array([1, 0, 0, 0])
    theta_norm = np.linalg.norm(theta)
    q = Quaternion(theta / theta_norm)    
    b = body.Body(np.zeros(3), q, r, 1)

    theta = np.array([1, 2.3, 0.1, 0.4])
    theta_norm = np.linalg.norm(theta)
    q = Quaternion(theta / theta_norm)
    r = b.get_r_vectors(orientation=q)
    r[:,2] += 3

  # Compute blob-blob distances
  x = r[:,0]
  y = r[:,1]
  z = r[:,2]
  dx = x - x[:,None]
  dy = y - y[:,None]
  dz = z - z[:,None]
  dr = np.sqrt(dx*dx + dy*dy + dz*dz)

  # Compute histogram distances and save it
  h, h_edges = np.histogram(dr.flatten(), x.size, range=(0, np.max(dr.flatten())), density=False)
  
  with open('create_soft_articulated_histogram.dat', 'w') as f:
    for i in range(x.size - 1):
      f.write(str(0.5*(h_edges[i] + h_edges[i+1])) + '  ' + str(h[i]) + '\n')

  # Print minim distance
  print('minimum distance = ', np.sort(dr.flatten())[x.size])

  # Loop over blobs and create links
  links = []
  if method == 'distance':
    for i in range(x.size-1):
      for j in range(i+1, x.size):
        if dr[i,j] <= d_max:
          l = np.zeros(8)
          l[0] = i
          l[1] = j
          l[2:5] = (r[j] - r[i]) * d_frac
          l[5:8] = (r[i] - r[j]) * (1 - d_frac)
          links.append(l)
  else:
    num_links = np.zeros(x.size)
    for i in range(x.size - 1):
      dri_ind  = np.argsort(dr[i])
      count = 0
      while num_links[i] < max_num_neighbors and count < x.size:
        j = dri_ind[count]
        if num_links[j] < max_num_neighbors and j > i:
          l = np.zeros(8)
          l[0] = i
          l[1] = j
          l[2:5] = (r[j] - r[i]) * d_frac
          l[5:8] = (r[i] - r[j]) * (1 - d_frac)
          links.append(l)
          num_links[i] += 1
          num_links[j] += 1
        count += 1
        
      


  # Save constraint file
  with open('create_soft_articulated.const', 'w') as f:
    f.write(str(x.size) + '\n')
    f.write(str(len(links)) + '\n')
    for l in links:
      f.write(str(int(l[0])) + ' ' + str(int(l[1])) + ' ' + str(l[2]) + ' ' + str(l[3]) + ' ' + str(l[4]) + ' ' + str(l[5]) + ' ' + str(l[6]) + ' ' + str(l[7]) + ' ' + '\n')

  # Save clones file
  with open('create_soft_articulated.clones', 'w') as f:
    f.write(str(x.size) + '\n')
    for ri in r:
      f.write(str(ri[0]) + ' ' + str(ri[1]) + ' ' + str(ri[2]) + ' 1 0 0 0 \n')

  # Save vertex list
  with open('create_soft_articulated.list_vertex', 'w') as f:
    for i in range(x.size):
      f.write('Structures/blob.vertex \n')

    
