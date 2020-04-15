import numpy as np
import sys

if __name__ == '__main__':

    # Set parameters
    Lg = 1.75
    Nx = 12

    # Calculate additional parameters
    dx = Lg / (Nx - 1)

    # Create rod
    r = np.zeros((Nx, 3))
    r[:,0] = np.linspace(-0.5 * Lg, 0.5 * Lg, Nx)

    # Print rod
    print Nx
    np.savetxt(sys.stdout, r, delimiter = '  ')
