''' Plot animation of the free tetrahedron trajectory. '''
import cPickle
import csv
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import animation
import numpy as np
import os
import sys
sys.path.append('..')

import tetrahedron_free as tf
from quaternion_integrator.quaternion import Quaternion
from config_local import DATA_DIR


if __name__ == '__main__':
  # Data file name where trajectory data is stored.
  data_file_name = 'free-tetrahedron-trajectory-dt-0.5-N-5000-testing-scheme-RFD.pkl'

  #######
  trajectory_data = dict()
  with open(os.path.join(
      DATA_DIR, 'tetrahedron', data_file_name), 'rb') as f:
    trajectory_data = cPickle.load(f)

  orientations = trajectory_data['orientation']
  locations = trajectory_data['location']

  fig = plt.figure()
  ax = Axes3D(fig) #fig.add_axes([0.1, 0.1, 0.8, 0.8])

  ax.set_xlim3d([-10., 10.])
  ax.set_ylim3d([-10., 10.])
  ax.set_zlim3d([-0.5, 8.])
  
  wall_x = [-10. + k*20./20. for k in range(20) ]
  for k in range(19):
    wall_x += [-10. + k*20./20. for k in range(20) ]

  wall_y = [-10. for _ in range(20)]
  for k in range(19):
    wall_y += [-10 + k*20./20. for _ in range(20)]

  blobs, = ax.plot([], [], [], 'bo', ms=24)
  wall, = ax.plot(wall_x, wall_y, np.zeros(400), 'k.')
  connectors = [0]*12
  for j in range(4):
    for k in range(j+1, 4):
      connector, = ax.plot([], [], [], 'b-', lw=2)
      connectors[j*3 + k] = connector

  def init_animation():
    ''' Initialize 3D animation.'''
    r_vectors = tf.get_free_r_vectors([0., 0., tf.H], Quaternion([1., 0., 0., 0.]))
    blobs.set_data([r_vectors[k][0] for k in range(len(r_vectors))], 
                   [r_vectors[k][1] for k in range(len(r_vectors))])
    blobs.set_3d_properties([r_vectors[k][2] for k in range(len(r_vectors))])
    for j in range(len(r_vectors)):
      for k in range(j+1, len(r_vectors)):
        connectors[j*3 + k].set_data([r_vectors[j][0],r_vectors[k][0]], 
                                     [r_vectors[j][1],r_vectors[k][1]])
        connectors[j*3 + k].set_3d_properties([r_vectors[j][2], r_vectors[k][2]])

    
  def update(n):
    ''' Update the tetrahedron animation '''
    location = locations[n]
    orientation = orientations[n]
    r_vectors = tf.get_free_r_vectors(location, Quaternion(orientation))
    blobs.set_data([r_vectors[k][0] for k in range(len(r_vectors))], 
                   [r_vectors[k][1] for k in range(len(r_vectors))])
    blobs.set_3d_properties([r_vectors[k][2] for k in range(len(r_vectors))])
    for j in range(len(r_vectors)):
      for k in range(j+1, len(r_vectors)):
        connectors[j*3 + k].set_data([r_vectors[j][0],r_vectors[k][0]], 
                                     [r_vectors[j][1],r_vectors[k][1]])
        connectors[j*3 + k].set_3d_properties([r_vectors[j][2], r_vectors[k][2]])
    
  
anim = animation.FuncAnimation(fig, update, init_func=init_animation, frames=500, interval=4, blit=True)
anim.save('tetrahedron.mp4', writer='ffmpeg')
