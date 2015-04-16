'''self mobility for a sphere close to a wall including stresslet. 
   Based on the paper J. W. Swan and J. F. Brady, Physics of fluids 19, 113306 (2007)
   and on L. Durlofsky, J. F. Brady and G. Bossis, J. Fluid Mech., 180, 21 (1987).

   Following the notation of Swan and Brady the full mobolity matrix is

   | M_UF  M_UL  M_US |
   |                  |
   | M_WF  M_WL  M_WS |
   |                  |
   | M_EF  M_EL  M_ES |

   sinse for us the rate of strein is zero we can rewrite the matrix like

   | (M_UF - M_US M_ES^{-1} M_EF)   (M_UL - M_US M_ES^{-1} M_EL) |
   |                                                             |
   | (M_WF - M_WS M_ES^{-1} M_EF)   (M_WL - M_WS M_ES^{-1} M_EL  |

   and this is the mobility that we want to write

   The wall is at z=0, the particle is assumed to be at z>=0.
   The mobility is normalize with 6*pi*eta*a^n
   where n keeps the terms dimensionless'''


import math
import numpy as np
import scipy 
import scipy.linalg as la




def selfMobilitySwanBradyWithStresslet(a, h):
    '''self mobility close to a wall inclusing stresslet.
       mobility components are normalize with 6*pi*eta*a^n
       where n keeps the terms dimensionless
       Based on the paper J. W. Swan and J. F. Brady Physics of fluids 19, 113306 (2007)
       and on L. Durlofsky, J. F. Brady  and G. Bossis, J. Fluid Mech., 180, 21 (1987).

       a = sphere radius
       h = distance between sphere center and plane
       '''
    
    #Normalize distance to the wall
    h = h / a

    II = np.identity(6)         # Identity matrix
    eijk = np.zeros((3, 3, 3))  # Levi-civita symbol
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] =  1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    mobility = np.zeros((6, 6)) # Matrix for translation and rotation
    


    #Build matrix M_ES
    #M_ES contribution far from the wall, from Durlofsky et al.
    M_ES = np.zeros( (5,5) )
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    index1 = i*3+j
                    index2 = k*3+l
                    if( (index1<=4 and index2<=4) or 1==0):
                        ii = i
                        jj = j
                        kk = k
                        ll = l
                        if(index1==3):
                            ii=1
                            jj=1
                        if(index1==4):
                            ii=1
                            jj=2
                        if(index2==3):
                            kk=1
                            ll=1
                        if(index2==4):
                            kk=1
                            ll=2
                        M_ES[index1, index2] = 1.35 * (II[ii,2]*II[jj,2] - (1.0/3.0)*II[ii,jj]) * (II[kk,2]*II[ll,2] - (1.0/3.0)*II[kk,ll]) \
                            + 0.45 * (II[ii,2]*II[jj,ll]*II[kk,2] + II[jj,2]*II[ii,ll]*II[kk,2] + II[ii,2]*II[jj,kk]*II[ll,2] \
                                          + II[jj,2]*II[ii,kk]*II[ll,2] - 4.0*II[ii,2]*II[jj,2]*II[kk,2]*II[ll,2] ) \
                                          + 0.45 *(II[ii,kk]*II[jj,ll] + II[jj,kk]*II[ii,ll] - II[ii,jj]*II[kk,ll] + II[ii,2]*II[jj,2]*II[kk,ll] \
                                                       + II[ii,jj]*II[kk,2]*II[ll,2] \
                                                       + II[ii,2]*II[jj,2]*II[kk,2]*II[ll,2] \
                                                       - II[ii,2]*II[jj,ll]*II[kk,2] \
                                                       - II[jj,2]*II[ii,ll]*II[kk,2] \
                                                       - II[ii,2]*II[jj,kk]*II[ll,2] \
                                                       - II[jj,2]*II[ii,kk]*II[ll,2])
                        

    #Build matrix M_ES
    #M_ES contribution close to the wall, from Swan and Brady
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    index1 = i*3+j
                    index2 = k*3+l
                    if( (index1<=4 and index2<=4) ):
                        ii = i
                        jj = j
                        ll = l
                        kk = k
                        if(index1==3):
                            ii=1
                            jj=1
                        if(index1==4):
                            ii=1
                            jj=2
                        if(index2==3):
                            kk=1
                            ll=1
                        if(index2==4):
                            kk=1
                            ll=2
                        M_ES[index1, index2] += \
                            -(3.0/640.0) * (10.0/h**3.0 - 24.0/h**5.0 + 9.0/h**7.0) * (II[ii,jj] - II[ii,2]*II[jj,2])*(II[kk,ll] - II[kk,2]*II[ll,2]) \
                            - (9.0/640.0) * (10.0/h**3.0 - 8.0/h**5.0 + 3.0/h**7.0) \
                            * ((II[ii,kk]-II[ii,2]*II[kk,2])*(II[jj,ll]-II[jj,2]*II[ll,2]) + (II[ii,ll]-II[ii,2]*II[ll,2])*(II[jj,kk]-II[jj,kk]*II[kk,2])) \
                            - (9.0/320.0) * (15.0/h**3.0 - 16.0/h**5.0 + 6.0/h**7.0) \
                            * ((II[ii,kk]-II[ii,2]*II[kk,2])*II[jj,2]*II[kk,2] + (II[ii,ll]-II[ii,2]*II[ll,2])*II[jj,2]*II[kk,2] \
                                   + (II[jj,kk]-II[jj,2]*II[kk,2])*II[ii,2]*II[ll,2] + (II[jj,ll]-II[jj,2]*II[ll,2])*II[ii,2]*II[kk,2]) \
                                   - (3.0/80.0) * (20.0/h**3.0 - 24.0/h**5.0 + 9.0/h**7.0) * II[ii,2]*II[jj,2]*II[kk,2]*II[ll,2] 
    #Invert M_ES
    M_ES_inv = la.inv(M_ES)


    #Build matrix M_UF
    # M_UF terms far from the wall
    M_UF = np.eye(3)

    # M_UF terms close to the wall
    for i in range(3):
        for j in range(i,3):
            M_UF[i,j] += \
            -(1.0/16.0) * (9.0/h - 2.0/(h**3.0) + 1.0/(h**5.0)) * (II[i][j] - II[i][2]*II[j][2]) \
            -(1.0/8.0) * (9.0/h - 4.0/(h**3.0) + 1.0/(h**5.0)) * II[i][2]*II[j][2]
            if i != j:
                M_UF[j,i] = M_UF[i,j]

    #Build matrix M_WL
    # M_WL terms far from the wall
    M_WL = 0.75 * np.eye(3)

    # M_WL terms from the wall
    for i in range(3):
        for j in range(i,3):
            M_WL[i,j] += \
                -(15.0/64.0) * (1.0/h**3.0) * (II[i][j] - II[i][2]*II[j][2]) \
                -(3.0/32.0) * (1.0/h**3.0) * II[i][2]*II[j][2]
            if i != j:
                M_WL[j,i] = M_WL[i,j]

    #Build matrix M_WF and M_UL
    #Terms far from the wall
    M_WF = np.zeros( (3,3) )
    # terms close to the wall
    for i in range(3):
        for j in range(3):
            M_WF[i,j] += -(3.0/32.0) * (1.0/(h**4.0)) * eijk[2][i][j] #CHECK SYMBOL!!!!
    M_UL = np.transpose(M_WF)

    #Build matrix M_US
    # M_EF terms far from the wall
    M_EF = np.zeros( (5,3) )

    # M_EF terms close to the wall
    for i in range(3):
        for j in range(3):
            for k in range(3):
                index = i*3 + j
                if(index==3):
                    ii=1
                    jj=1
                if(index==4):
                    ii=1
                    jj=2
                if(index <= 4): #We only need the elements for the 5x3 matrix
                    M_EF[index, k] = -(3.0/160.0) * (15.0/h**2.0 - 12.0/h**4.0 + 5.0/h**6.0) \
                        * ( (II[ii,k]-II[ii,2]*II[k,2]) * II[jj,2]  +  (II[jj,k]-II[jj,2]*II[k,2]) * II[ii,2] ) \
                        + (3.0/32.0) * (3.0/h**2.0 - 3.0/h**4.0 + 1.0/h**6.0) \
                        * (II[ii,jj] - II[ii,2]*II[jj,2])*II[k,2] \
                        - (3.0/16.0) * (3.0/h**2.0 -3.0/h**4.0 + 1.0/h**6.0) \
                        * II[ii,2] * II[jj,2] * II[k,2]           

    # Build matrix M_US
    M_US = np.transpose(M_EF)

    # Build matrix M_EL
    # M_EL terms far from the wall
    M_EL = np.zeros( (5,3) )
    
    # M_EL terms close to the wall
    for i in range(3):
        for j in range(3):
            for k in range(3):
                index = i*3 + j
                if(index==3):
                    ii=1
                    jj=1
                if(index==4):
                    ii=1
                    jj=2
                if(index <= 4): #We only need the elements for the 5x3 matrix
                    M_EL[index, k] = -(9.0/320.0) * (5.0/h**3.0 - 4.0/h**5.0) * (II[jj,2]*eijk[2,ii,k] + II[ii,2]*eijk[2,jj,k])
    
    # Build matrix M_WS
    M_WS = np.transpose(M_EL)

    #print np.dot(matrix_inv, matrix)
    #print "M_ES \n ", M_ES
    #print "M_ES_inv \n ", M_ES_inv
    #print "II \n", np.dot(M_ES, M_ES_inv)
    #print "II \n", np.dot(M_ES_inv, M_ES)
    
    M_UF_stresslet = M_UF - np.dot(M_US, np.dot(M_ES_inv, M_EF))
    M_UL_stresslet = M_UL - np.dot(M_US, np.dot(M_ES_inv, M_EL))
    M_WF_stresslet = M_WF - np.dot(M_WS, np.dot(M_ES_inv, M_EF))
    M_WL_stresslet = M_WL - np.dot(M_WS, np.dot(M_ES_inv, M_EL))

    #mobility = [ [M_UF_stresslet, M_UL_stresslet] , [M_WF_stresslet, M_WL_stresslet] ]
    mobility = np.concatenate( (np.concatenate( (M_UF_stresslet, M_UL_stresslet) , axis=1), np.concatenate( (M_WF_stresslet, M_WL_stresslet) , axis=1)) )


    return mobility






if __name__ == '__main__':
    #Define parameters
    eta = 1.0 #viscosity
    a = 1.0   #radius
    h = 3.0   #distance sphere center to wall

    mobility = selfMobilitySwanBradyWithStresslet(a, h)

    print 'mobility \n', mobility
