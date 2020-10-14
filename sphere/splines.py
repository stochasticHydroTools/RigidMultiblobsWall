'''This code uses cubic splines from to interpolate a parametrized
   funtion to an arbitrary point x inside the domain.
   
   It follows the codes in the book 'Numerical recipes for C'
   '''

import numpy as np


def spline(x, y, n, yp0, ypn_1):
    '''Given arrays x[0,n-1] and y[0,n-1] containing a tabulated function, i.e., yi = f(xi), with
    x0 <x1 < :: : < x(N-1), and given values yp0 and ypn_1 for the first derivative of the interpolating
    function at points 0 and n-1, respectively, this routine returns an array y2[0, n-1] that contains
    the second derivatives of the interpolating function at the tabulated points xi. If yp0 and/or
    ypn_1 are equal to 1e+30 or larger, the routine is signaled to set the corresponding boundary
    condition for a natural spline, with zero second derivative on that boundary.
    '''
    
    # Create vectors
    u = np.zeros(n-1)
    y2 = np.zeros(n)

    # Set lower boundary condition
    if(yp0 > 0.99e+30):
        y2[0] = 0.
        u[0] = 0.
    else:
        y2[0] = -0.5
        u[0] = (3.0/(x[1]-x[0])) * ((y[1]-y[0])/(x[1]-x[0]) - yp0)
    
    # Decomposition loop
    for i in range(1,n-1):
        sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1])
        p = sig * y2[i-1] + 2.0
        y2[i] = (sig-1.0) / p
        u[i] = (y[i+1]-y[i]) / (x[i+1]-x[i]) - (y[i]-y[i-1]) / (x[i]-x[i-1])
        u[i] = ( 6.0*u[i]/(x[i+1]-x[i-1]) - sig*u[i-1] ) / p
        #u[i] = (6.0 * ((y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1]) / (x[i]-x[i-1]))/(x[i+1]-x[i-1])-sig*u[i-1]) / p
    
    # Set upper boundary condition
    if(ypn_1 > 0.99e+30):
        qn_1 = 0.0
        un_1 = 0.0
    else:
        qn_1 = 0.5
        un_1 = (3.0/(x[n-1]-x[n-2])) * (ypn_1 - (y[n-1]-y[n-2])/(x[n-1]-x[n-2]))

    # Backsustitution algorithm
    y2[n-1] = (un_1 - qn_1*u[n-2]) / (qn_1*y2[n-2] + 1.0)
    for i in range(n-2,-1,-1):
        y2[i] = y2[i]*y2[i+1] + u[i]

    return y2







def splint(xa, ya, y2a, n, x):
    '''Given the arrays xa[0, n-1] and ya[0, n-1], which tabulate a function (with the xai's in order),
    and given the array y2a[0, n-1], which is the output from spline above, and given a value of
    x, this routine returns a cubic-spline interpolated value y.
    '''
    klo = 0
    khi = n-1
    # Find the right place in the table
    while(khi-klo > 1):
        k = (khi + klo) >> 1
        if(xa[k] > x):
            khi=k
        else:
            klo=k
            
    h = xa[khi] - xa[klo]
    if(h == 0):
        print('Bad xa input to routine splint')
        return 1e+999
    a = (xa[khi]-x) / h
    b = (x-xa[klo]) / h
    y = a*ya[klo] + b*ya[khi] + ((a**3-a)*y2a[klo] + (b**3-b)*y2a[khi])*(h**2) / 6.0

    return y





