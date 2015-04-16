'''compute the self mobility for a sphere between two walls
   using the MCSA theory and the mobility for a single wall
   based on a infinite (truncate) series. 

   To see the MCSA formula see paper
   S. Delong et al. The Journal of Chemical Physics 140, 134110 (2014).

   The mobility is normalize by 6*pi*eta*a
   '''

import selfMobilityBrenner
import selfMobilityFauxcheux
import selfMobilityHuang
import selfMobilityGoldman


def MCSA(a, h, L, termsMCSA, termsBrenner):
    '''compute the self mobility for a sphere between two walls
    using the MCSA theory and the mobility for a single wall
    based on a infinite (truncate) series. 
    
    To see the MCSA formula see paper
    S. Delong et al. The Journal of Chemical Physics 140, 134110 (2014).
    

    a = sphere radius
    h = distance between sphere center and plane
    L = distance between walls
    termsMCSA = number of terms in the MCSA serie 
    termsBrenner = number of terms in the Brenner Mobility

    The mobility is normalize by 6*pi*eta*a
    '''
    
    mobility = [0, 0]

    #Perpendicular mobility for one wall
    mu = [1, 1]
    mu1 = [0, 0]
    for n in range(0, termsMCSA):
        #Distance to the wall with reflections
        distance = n*L + h
        gap = (distance - a) / a
        z = a / distance
        #Parallel mobility
        if(gap > 0.05): #Use Goldman mobility
            mu1[0] = (1.0 - (9.0/16.0)*z + 0.125*z**3 - (45.0/256.0)*z**4 - 0.0625*z**5)
        else: #Use Fauxcheux mobility
            mu1[0] = (selfMobilityGoldman.selfMobilityGoldman(1.0, distance))[0][0]
        mu[0] += ((-1)**n)  *  (1.0/mu1[0] - 1)
        #Perpendicular mobility
        mu1[1] = (6.0 * gap**2 + 2.0 * gap) / (6.0 * gap**2 + 9*gap + 2.0)
        mu[1] += ((-1)**n)  *  (1.0/mu1[1] - 1)

        #Distance to the wall with reflections
        distance = (n+1)*L - h
        gap = (distance - a) / a
        z = a / distance
        #Parallel mobility
        if(gap > 0.05): #Use Goldman mobility
            mu1[0] = (1.0 - (9.0/16.0)*z + 0.125*z**3 - (45.0/256.0)*z**4 - 0.0625*z**5)
        else: #Use Fauxcheux mobility
            mu1[0] = (selfMobilityGoldman.selfMobilityGoldman(1.0, distance))[0][0]
        mu[0] += ((-1)**n)  *  (1.0/mu1[0] - 1)
        #Perpendicular mobility
        mu1[1] = (6.0 * gap**2 + 2.0 * gap) / (6.0 * gap**2 + 9*gap + 2.0)
        mu[1] += ((-1)**n)  *  (1.0/mu1[1] - 1)
        

    #Parallel mobility
    mobility[0] = 1.0 / mu[0]
    #Perpendicular mobility
    mobility[1] = 1.0 / mu[1]

    
    return mobility

if __name__ == '__main__':
    #Define parameters
    eta = 1.0 #viscosity
    a = 1.0   #radius
    h = 1.00000001   #distance sphere center to wall
    L = 50.9960159363
    n = 10
    m = 64
    dh = 0.01


    for i in range(0, 70):
        #if(h > 1.1):
        dh = dh * 1.08
        h += dh

        mobility = MCSA(a, h, L, n, m)
        print h, mobility[0], mobility[1]
