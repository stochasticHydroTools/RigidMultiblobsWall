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
  data_file_name = 'free-tetrahedron-trajectory-dt-0.1-N-3000-testing-scheme-FIXMAN.pkl'

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
  
  wall_x = [-10. + k*20./40. for k in range(40) ]
  for k in range(39):
    wall_x += [-10. + k*20./40. for k in range(40) ]

  wall_y = [-10. for _ in range(40)]
  for k in range(39):
    wall_y += [-10 + k*20./40. for _ in range(40)]

  blobs, = ax.plot([], [], [], 'bo')
  wall, = ax.plot(wall_x, wall_y, np.zeros(1600), 'k-')
  connectors, = ax.plot([], [], 'b-', lw=2)

  def init_animation():
    ''' Initialize 3D animation.'''
    r_vectors = tf.get_free_r_vectors([0., 0., tf.H], Quaternion([1., 0., 0., 0.]))
    blobs.set_data([r_vectors[k][0] for k in range(len(r_vectors))], 
                   [r_vectors[k][1] for k in range(len(r_vectors))])
    blobs.set_3d_properties([r_vectors[k][2] for k in range(len(r_vectors))])
    for j in range(len(r_vectors)):
      for k in range(j+1, len(r_vectors)):
        

    
  def update(n):
    ''' Update the tetrahedron animation '''
    location = locations[n]
    orientation = orientations[n]
    r_vectors = tf.get_free_r_vectors(location, Quaternion(orientation))
    blobs.set_data([r_vectors[k][0] for k in range(len(r_vectors))], 
                   [r_vectors[k][1] for k in range(len(r_vectors))])
    blobs.set_3d_properties([r_vectors[k][2] for k in range(len(r_vectors))])
    
  
anim = animation.FuncAnimation(fig, update, init_func=init_animation, frames=500, interval=7, blit=True)
anim.save('tetrahedron.mp4', writer='ffmpeg')
