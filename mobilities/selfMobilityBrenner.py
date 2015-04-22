'''compute the self mobility for a sphere close to a wall
   using an infinite (truncate) series. Based on the paper
   H. Brenner, Chemical engineering Science, 16, 242 (1961).

   The mobility is normalize by 6*pi*eta*a
   '''


import math
import numpy as np


def selfMobilityBrenner(a, h, n):
    '''compute the self mobility for a sphere close to a wall
    using an infinite (truncate) series. Based on the paper
    H. Brenner, Chemical engineering Science, 16, 242 (1961).
    
    The mobility is normalize by 6*pi*eta*a
    
    a = sphere radius
    h = distance between sphere center and plane
    n = number of terms in the serie
    friction = normalized friction, Eq. 2.19'''
    
    
    #alpha Eq. 1.5
    alpha = np.log(h/a + np.sqrt((h/a)**2 - 1.0))

    #Friction from Eq. 2.19
    friction = 0

    for i in range(1, n):
        
        #print "Brenner", i, alpha
        s1 = 2.0 * math.sinh( (2.0*i+1.0) * alpha)
        s2 = (2.0*i+1.0) * math.sinh( 2.0 * alpha)
        s3 = 4.0 * (math.sinh( (i+0.5) * alpha))**2
        s4 = ( (2.0*i+1.0) * math.sinh(alpha) )**2 
        
        friction += (i*(i+1.0) / ((2.0*i-1.0) * (2.0*i+3.0))) * ( (s1+s2)/(s3-s4) - 1.0)

    friction *= (4.0/3.0) * math.sinh(alpha)
    
    #Normalize mobility
    mobility = 1.0 / friction

    return mobility

if __name__ == '__main__':
    #Define parameters
    eta = 1.0 #viscosity
    a = 1.0   #radius
    h = 2.0   #distance sphere center to wall
    n = 10

    mobility = selfMobilityBrenner(a, h, n)

    print 'mobility \n', mobility
