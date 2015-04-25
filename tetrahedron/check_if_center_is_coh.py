''' Check if the geometric center of the tetrahedron is a CoH or CoD in bulk.'''


import sys
sys.path.append('..')

from quaternion_integrator.quaternion import Quaternion
import tetrahedron_free as tf

if __name__ == '__main__':
  
  location = [0., 0., 9000000.]
  orientation = Quaternion([1., 0., 0., 0.])

  mobility = tf.free_tetrahedron_center_mobility([location], [orientation])

  print "mobility coupling is : ", mobility[0:3, 3:6]

