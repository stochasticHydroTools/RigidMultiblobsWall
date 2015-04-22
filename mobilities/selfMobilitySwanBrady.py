'''self mobility for a sphere close to a wall. Based on the paper
   J. W. Swan and J. F. Brady Physics of fluids 19, 113306 (2007)

   The wall is at z=0, the particle is assumed to be at z>=0.
   The mobility is normalize with 6*pi*eta*a^n
   where n keeps the terms dimensionless'''


import math
import numpy as np





def selfMobilitySwanBrady(a, h):
    '''self mobility close to a wall.
       mobility components are normalize with 6*pi*eta*a^n
       where n keeps the terms dimensionless

       a = sphere radius
       h = distance between sphere center and plane
       '''
    
    #Normalize distance to the wall
    h = h / a

    II = np.identity(6)         #Identity matrix
    eijk = np.zeros((3, 3, 3))  #Levi-civita symbol
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    mobility = np.zeros((6, 6)) #Matrix for translation and rotation
    
    #Add terms far from the wall
    for i in range(3):
        mobility[i][i] += 1            #translation
        mobility[i+3][i+3] += 3.0/4.0  #rotation
        

    #Add terms close to the wall
    #M_UF
    for i in range(3):
        for j in range(i,3):
            mobility[i][j] += \
            -(1.0/16.0) * (9.0/h - 2.0/(h**3.0) + 1.0/(h**5.0)) * (II[i][j] - II[i][2]*II[j][2]) \
            -(1.0/8.0) * (9.0/h - 4.0/(h**3.0) + 1.0/(h**5.0)) * II[i][2]*II[j][2]
            if i != j:
                mobility[j][i] = mobility[i][j]


    #M_WT
    for i in range(3,6):
        for j in range(i,6):
            mobility[i][j] += \
                -(15.0/64.0) * (1.0/h**3.0) * (II[i][j] - II[i][5]*II[j][5]) \
                -(3.0/32.0) * (1.0/h**3.0) * II[i][5]*II[j][5]
            if i != j:
                mobility[j][i] = mobility[i][j]

    #M_WF and M_UT
    for i in range(3):
        for j in range(3):
            mobility[3+i][j] += -(3.0/32.0) * (1.0/(h**4.0)) * eijk[2][i][j] #CHECK SYMBOL!!!!
            mobility[j][3+i] = mobility[3+i][j]

    return mobility






if __name__ == '__main__':
    #Define parameters
    eta = 1.0 #viscosity
    a = 1.0   #radius
    h = 2.0   #distance sphere center to wall

    mobility = selfMobilitySwanBrady(a, h)

    print 'mobility \n', mobility
