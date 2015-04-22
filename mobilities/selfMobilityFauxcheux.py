'''compute the self mobility for a sphere close to a wall
   using a expansion accurate to order (a/h)**5.
   Based on the paper L. P. Fauxcheux and A. J. Libchaber
   PRE 49, 5158 (1994)

   The mobility is normalize by 6*pi*eta*a
   '''

import math
import numpy as np


def selfMobilityFauxcheux(a, h):
    '''compute the self mobility for a sphere close to a wall
    using a expansion accurate to order (a/h)**5
    Based on the paper L. P. Fauxcheux and A. J. Libchaber
    PRE 49, 5158 (1994)
    
    This code computes the parallel mobility. For
    the perpendicular mobility use selfMobilityBrenner
    
    The mobility is normalize by 6*pi*eta*a
    a = radius 
    h = distance between sphere center and wall
    '''
    
    #Normalize and invert distance to the wall
    z = a / h

    mobility = (1.0 - (9.0/16.0)*z + 0.125*z**3 - (45.0/256.0)*z**4 - 0.0625*z**5)
    
    return mobility







if __name__ == '__main__':
    #Define parameters
    eta = 1.0 #viscosity
    a = 1.0   #radius
    h = 2.0   #distance sphere center to wall

    mobility = selfMobilityFauxcheux(a, h)

    print 'mobility \n', mobility
