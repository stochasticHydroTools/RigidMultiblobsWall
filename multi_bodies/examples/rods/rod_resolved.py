from __future__ import division, print_function
import numpy as np
import sys

if __name__ == '__main__':
    
    # Set parameters
    # escale_factor = 0.900615163
    escale_factor = 0.977702385
    Lg = 2.5
    Rg = 0.15
    Nx = 72
    Ntheta = 24
    caps_layers = 4
    alpha = 2.5 / 0.3
    a = Rg * np.sin(np.pi / Ntheta)
    pattern = 'hexagonal' # square or hexagonal pattern

    # Scale Lg, Rg and a
    Rg = Rg * escale_factor
    a  = a * escale_factor
    if pattern == 'hexagonal':
        # Lg = escale_factor * (alpha * (2 * Rg + a) - a)
        # Lg = (alpha * (2*Rg + a) - a)
        Lg = escale_factor * Lg
    elif pattern == 'square':
        Lg = Lg * escale_factor
        
    print('# a  = ', a)
    print('# Rg = ', Rg)
    print('# Lg = ', Lg)
    print('# caps_layers = ', caps_layers)

    # Calculate additional parameters
    dx = Lg / (Nx - 1)
    dtheta = 2 * np.pi / Ntheta
    dcaps = Rg / np.maximum(caps_layers, 1)
    
    # Count caps
    Ncaps = 0
    for ilayer in range(caps_layers):
        perimeter = 2 * np.pi * ilayer * dcaps
        Ntheta_layer = int(np.floor((perimeter / (2 * np.pi * Rg)) * Ntheta)) + 1
        for itheta in range(Ntheta_layer):
            Ncaps += 2


    # Print rod sides
    r_all = np.zeros((Nx * Ntheta + Ncaps, 3))
    count = 0
    for ix in range(Nx):
        for itheta in range(Ntheta):
            rx = ix * dx - Lg * 0.5

            if pattern == 'hexagonal':
                theta = dtheta * (itheta + 0.5 * (ix % 2))
            elif pattern == 'square':
                theta = dtheta * itheta
            ry = Rg * np.cos(theta)
            rz = Rg * np.sin(theta)

            r = np.array([rx, ry, rz])
            # np.savetxt(sys.stdout, r[None, :], delimiter = '  ')
            r_all[count] = r
            count += 1


    # Print rod caps 
    for ilayer in range(caps_layers):
        perimeter = 2 * np.pi * ilayer * dcaps
        Ntheta_layer = int(np.floor((perimeter / (2 * np.pi * Rg)) * Ntheta)) + 1
        dtheta_layer = 2 * np.pi / np.maximum(Ntheta_layer, 1)

        # One cap
        rx = -Lg * 0.5
        for itheta in range(Ntheta_layer):
            theta = dtheta_layer * (itheta + 0.5 * (ilayer % 2))
            ry = dcaps * ilayer * np.cos(theta)
            rz = dcaps * ilayer * np.sin(theta)
            
            r = np.array([rx, ry, rz])
            # np.savetxt(sys.stdout, r[None, :], delimiter = '  ')
            r_all[count] = r
            count += 1
            
        # Other cap
        rx = Lg * 0.5
        for itheta in range(Ntheta_layer):
            theta = dtheta_layer * (itheta + 0.5 * (ilayer % 2))
            ry = dcaps * ilayer * np.cos(theta)
            rz = dcaps * ilayer * np.sin(theta)
            
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
    print(Nx * Ntheta + Ncaps)
    np.savetxt(sys.stdout, r_all, delimiter = '  ')
