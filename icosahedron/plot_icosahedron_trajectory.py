''' 
Plot animation of the nonuniform icosahedron. 
Can also be used for uniform icosahedron, but this script
will color the 12th vertex differently becuase this is the 
vertex with all of the mass in the nonuniform case.
'''

import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import animation
import numpy as np
import os
import sys
sys.path.append('..')

from config_local import DATA_DIR
import icosahedron as ic
import icosahedron_nonuniform as icn
from quaternion_integrator.quaternion import Quaternion
from utils import read_trajectory_from_txt

if __name__ == '__main__':
  # Data file of trajectory data.
  data_name = sys.argv[1]
  data_file_name = os.path.join(DATA_DIR, 'icosahedron', data_name)

  # Read trajectory data.
  params, locations, orientations = read_trajectory_from_txt(data_file_name)
  fig = plt.figure()
  ax = Axes3D(fig) #fig.add_axes([0.1, 0.1, 0.8, 0.8])

  ax.set_xlim3d([-3., 3.])
  ax.set_ylim3d([-3., 3.])
  ax.set_zlim3d([-0.5, 5.5])
  
  wall_x = [-3. + k*6./20. for k in range(20) ]
  for k in range(19):
    wall_x += [-3. + k*6./20. for k in range(20) ]

  wall_y = [-3. for _ in range(20)]
  for k in range(19):
    wall_y += [-3. + k*6./20. for _ in range(20)]


  wall, = ax.plot(wall_x, wall_y, np.zeros(400), 'k.')
  blobs, = ax.plot([], [], [], 'bo', ms=12)
  last_blob, = ax.plot([], [], [], 'ro', ms=12)
  connectors = [0]*12*11
  for j in range(12):
    for k in range(j+1, 12):
      pass
      connector, = ax.plot([], [], [], 'b-', lw=1)
      connectors[j*3 + k] = connector
  
  def init_animation():
    ''' Initialize 3D animation.'''
    r_vectors = ic.get_icosahedron_r_vectors(
      locations[0], Quaternion(orientations[0]))
    blobs.set_data([r_vectors[k][0] for k in range(len(r_vectors) -1)], 
                   [r_vectors[k][1] for k in range(len(r_vectors) - 1)])
    blobs.set_3d_properties([r_vectors[k][2] for k in range(len(r_vectors) - 1)])

    last_blob.set_data([r_vectors[-1][0]],
                       [r_vectors[-1][1]])
    last_blob.set_3d_properties([r_vectors[-1][2]])
    for j in range(len(r_vectors)):
      for k in range(j+1, len(r_vectors)):
        connectors[j*3 + k].set_data([r_vectors[j][0],r_vectors[k][0]], 
                                     [r_vectors[j][1],r_vectors[k][1]])
        connectors[j*3 + k].set_3d_properties([r_vectors[j][2], r_vectors[k][2]])

  def update(n):
    ''' Update the icosahedron animation '''
    location = locations[n]
    orientation = orientations[n]
    r_vectors = ic.get_icosahedron_r_vectors(
      location, Quaternion(orientation))
    blobs.set_data([r_vectors[k][0] for k in range(len(r_vectors))], 
                   [r_vectors[k][1] for k in range(len(r_vectors))])
    blobs.set_3d_properties([r_vectors[k][2] for k in range(len(r_vectors))])

    last_blob.set_data([r_vectors[-1][0]],
                       [r_vectors[-1][1]])
    last_blob.set_3d_properties([r_vectors[-1][2]])
    for j in range(len(r_vectors)):
      for k in range(j+1, len(r_vectors)):
        connectors[j*3 + k].set_data([r_vectors[j][0],r_vectors[k][0]], 
                                     [r_vectors[j][1],r_vectors[k][1]])
        connectors[j*3 + k].set_3d_properties([r_vectors[j][2], r_vectors[k][2]])
    
  
anim = animation.FuncAnimation(fig, update, init_func=init_animation, 
                               frames=400, interval=1, blit=True)
anim.save('./figures/icosahedron.mp4', fps=20, writer='ffmpeg')



