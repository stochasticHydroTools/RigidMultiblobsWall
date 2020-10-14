'''Check if the new sphere mobility is positive definite for all distance'''



import numpy as np
import sys
from . import sphere as sph
sys.path.append('..')




from fluids import mobility as mb
from quaternion_integrator.quaternion import Quaternion





if __name__ == '__main__':

    # Parameters
    points = 1000
    distance = sph.A * 2
    orientation = Quaternion([1., 0., 0., 0.])

    location = [ [0., 0., distance] ]
    dd = (distance - sph.A * 0.9) / float(points)
    distance = distance + dd
    

    # Loop for distances
    if(1):
        for i in range(points):
            distance -= dd
            #print i, distance
            location = [ [0., 0., distance] ]
            mobility = sph.sphere_mobility(location, orientation)
            data = str(distance/sph.A) + '  '
            data += str(mobility[0, 0] * (6.0*np.pi*sph.ETA * sph.A)) + '  '
            data += str(mobility[2, 2] * (6.0*np.pi*sph.ETA * sph.A)) + '  '
            data += str(mobility[3, 3] * (8.0*np.pi*sph.ETA * sph.A**3)) + '  '
            data += str(mobility[5, 5] * (8.0*np.pi*sph.ETA * sph.A**3)) + '  '
            data += str(mobility[0, 4] * (6.0*np.pi*sph.ETA * sph.A**2)) 
            print(data)
            mobility_half = np.linalg.cholesky(mobility)


    print("#END")
