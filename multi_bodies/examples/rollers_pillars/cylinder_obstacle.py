from __future__ import division, print_function
import numpy as np
import sys

if __name__ == '__main__':
    
    # Set parameters
    # escale_factor = 0.900615163
    escale_factor = 1
    Rg = 0.5
    Nz = 15
    Ntheta = 8
    a = Rg * np.sin(np.pi / Ntheta)
    pattern = 'hexagonal' # square or hexagonal pattern

    # Scale Lg, Rg and a
    Rg = Rg * escale_factor
    a  = a * escale_factor
    if pattern == 'hexagonal':
        dz = np.sqrt(3.0)*a
    elif pattern == 'square':
        dz = 2.0*a
    Lg = (Nz-1)*dz 
        
    print('# a  = ', a)
    print('# Rg = ', Rg)
    print('# Lg = ', Lg)

    # Calculate additional parameters
    dtheta = 2 * np.pi / Ntheta
    


    # Print rod sides
    r_all = np.zeros((Nz * Ntheta , 3))
    count = 0
    for iz in range(Nz):
        for itheta in range(Ntheta):
            rz = iz * dz + a*0.5

            if pattern == 'hexagonal':
                theta = dtheta * (itheta + 0.5 * (iz % 2))
            elif pattern == 'square':
                theta = dtheta * itheta
            rx = Rg * np.cos(theta)
            ry = Rg * np.sin(theta)

            r = np.array([rx, ry, rz])
            # np.savetxt(sys.stdout, r[None, :], delimiter = '  ')
            r_all[count] = r
            count += 1


            
    # Compute shortest distance between blobs
    x = r_all[:,0]
    y = r_all[:,1]
    z = r_all[:,2]
    dx = x - x[:,None]
    dy = y - y[:,None]
    dz = z - z[:,None]
    dr = np.sqrt(dx**2 + dy**2 + dz**2)
    np.fill_diagonal(dr, 1e+99)
    dr = dr.flatten()
    dr_min = np.min(dr)
    print('# Minimum distance between blobs = ', dr_min)

    # Print coordinates
    print(Nz * Ntheta)
    np.savetxt(sys.stdout, r_all, delimiter = '  ')
