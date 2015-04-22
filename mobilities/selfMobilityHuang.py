'''compute the self mobility for a sphere close to a wall
   using approximation valid close to the wall.
   Based on the paper P. Haung and K. S. Breuer
   PRE 76, 046307 (2007).

   The mobility is normalize by 6*pi*eta*a
   '''


import math
import numpy as np


def selfMobilityHuang(a, h):
    '''compute the self mobility for a sphere close to a wall
    using approximation valid close to the wall.
    Based on the paper P. Haung and K. S. Breuer
    PRE 76, 046307 (2007).
    
    The mobility is normalize by 6*pi*eta*a
    
    a = sphere radius
    h = distance between sphere center and plane
    '''
    
    #Normalize gap between sphere and wall
    gap = (h - a) / a

    #Parallel and perpendicular mobilities
    mobility = [0, 0] 

    #Parallel mobility
    mobility[0] = - (2.0 * (math.log(gap) - 0.9543)) / ( (math.log(gap))**2 - 4.325*math.log(gap) + 1.591)
    #mobility[0] = - (2.0 * (math.log(gap) - 0.95425)) / ( (math.log(gap))**2 - 3.1881*math.log(gap) + 1.5905313)

    #Perpendicular mobility
    mobility[1] = (6.0 * gap**2 + 2.0 * gap) / (6.0 * gap**2 + 9*gap + 2.0)

    return mobility


if __name__ == '__main__':
    #Define parameters
    eta = 1.0 #viscosity
    a = 1.0   #radius
    h = 2.0   #distance sphere center to wall
    

    mobility = selfMobilityHuang(a, h)

    print 'mobility \n', mobility
