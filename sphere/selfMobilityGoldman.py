'''compute the self mobility for a sphere close to a wall
   using approximation valid close to the wall.
   A. J. Goldman, R. G. Cox and H. Brenner,
   Chemical engineering science, 22, 637 (1967).

   The mobility is normalize by 6*pi*eta*a
   '''


import math
import numpy as np



def selfMobilityGoldman(a, h):
    '''compute the self mobility for a sphere close to a wall
    using approximation valid close to the wall.
    A. J. Goldman, R. G. Cox and H. Brenner,
    Chemical engineering science, 22, 637 (1967).    
    The mobility is normalize by 6*pi*eta*a
    
    a = sphere radius
    h = distance between sphere center and plane
    '''
    
    #Normalize gap between sphere and wall
    gap = (h - a) / a
    if(gap < 0):
        gap = - gap

    #Factors to use in the resistance
    factor_FT = 6*np.pi*a
    factor_FR = 6*np.pi*a*2
    factor_TT = 8*np.pi*a**2
    factor_TR = 8*np.pi*a**3

    #Resistance
    resistance = np.zeros((2,2))
    
    #Component R_FT (T for translation)
    resistance[0][0] = factor_FT * ((8.0/15.0) * math.log(gap) - 0.9588) #Eq. 2.65
    #Component R_FR (R for rotation)
    resistance[0][1] = factor_FR * (-(2.0/15.0)*math.log(gap) - 0.2526)  #Eq. 3.13
    #Component R_TT
    resistance[1][0] = factor_TT * (-0.1*math.log(gap) - 0.1895)         #Eq. 2.65
    #Component R_TR
    resistance[1][1] = factor_TR * (0.4*math.log(gap) - 0.3817)          #Eq. 3.13

    determinant = resistance[0][0]*resistance[1][1] - resistance[0][1]*resistance[1][0]

    #Mobility
    mobility = np.zeros((2,2))
    mobility[0][0] =  resistance[1][1] / determinant
    mobility[0][1] = -resistance[0][1] / determinant
    mobility[1][0] = -resistance[1][0] / determinant
    mobility[1][1] =  resistance[0][0] / determinant

    mobility = -1 * (6.0*np.pi*1.0*a) * mobility

    return mobility


