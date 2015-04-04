''' Plot animation of the Boomerang. '''

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import animation
import numpy as np
import os
import sys
sys.path.append('..')

import boomerang as bm
from config_local import DATA_DIR
from quaternion_integrator.quaternion import Quaternion
from utils import read_trajectory_from_txt


if __name__ == '__main__':
  # Data file name where trajectory data is stored.
  data_name = sys.argv[1]


  x_lim = [-5., 5.]
  y_lim = [-5., 5.]
  z_lim = [-0.5, 9.5]

  #######
  data_file_name = os.path.join(DATA_DIR, 'boomerang', data_name)
  
  params, locations, orientations = read_trajectory_from_txt(data_file_name)

  print 'Parameters are : ', params

  fig = plt.figure()
  ax = Axes3D(fig)
  ax.set_xlim3d(x_lim)
  ax.set_ylim3d(y_lim)
  ax.set_zlim3d(z_lim)
  
  wall_x = [x_lim[0] + k*(x_lim[1] - x_lim[0])/20. for k in range(20) ]
  for k in range(19):
    wall_x += [x_lim[0] + k*(x_lim[1] - x_lim[0])/20. for k in range(20) ]

  wall_y = [y_lim[0] for _ in range(20)]
  for k in range(19):
    wall_y += [y_lim[0] + k*(y_lim[1] - y_lim[0])/20. for _ in range(20)]


  wall, = ax.plot(wall_x, wall_y, np.zeros(400), 'k.')
  blobs, = ax.plot([], [], [], 'bo', ms=24)
  connectors, = ax.plot([], [], [], 'b-', lw=2)


  def init_animation():
    ''' Initialize 3D animation. '''
    r_vectors = bm.get_boomerang_r_vectors(locations[0], 
                                           Quaternion(orientations[0]))
    blobs.set_data([r_vectors[k][0] for k in range(len(r_vectors))], 
                   [r_vectors[k][1] for k in range(len(r_vectors))])
    blobs.set_3d_properties([r_vectors[k][2] for k in range(len(r_vectors))])
    connectors.set_data([r_vectors[k][0] for k in range(len(r_vectors))],
                       [r_vectors[k][1] for k in range(len(r_vectors))])
    connectors.set_3d_properties([r_vectors[k][2] for k in range(len(r_vectors))])

    
  def update(n):
    ''' Update the boomerang animation '''
    location = locations[n]
    orientation = orientations[n]
    r_vectors = bm.get_boomerang_r_vectors(location, 
                                           Quaternion(orientation))
    blobs.set_data([r_vectors[k][0] for k in range(len(r_vectors))], 
                   [r_vectors[k][1] for k in range(len(r_vectors))])
    blobs.set_3d_properties([r_vectors[k][2] for k in range(len(r_vectors))])
    connectors.set_data([r_vectors[k][0] for k in range(len(r_vectors))],
                       [r_vectors[k][1] for k in range(len(r_vectors))])
    connectors.set_3d_properties([r_vectors[k][2] for k in range(len(r_vectors))])
    
  
anim = animation.FuncAnimation(fig, update, init_func=init_animation, 
                               frames=800, interval=2, blit=True)
anim.save(os.path.join('figures','boomerang.mp4'), fps=30, writer='ffmpeg')

