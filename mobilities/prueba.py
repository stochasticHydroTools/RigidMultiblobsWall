
import numpy as np
import scipy
import scipy.linalg as la



if __name__ == '__main__':

    M = np.zeros( (5,5) )
    M = np.random.rand(5,5)
    b = np.random.rand(5)

    x = la.solve(M,b, overwrite_a=1, overwrite_b=1)
                                                   

    print "hola \n", x
