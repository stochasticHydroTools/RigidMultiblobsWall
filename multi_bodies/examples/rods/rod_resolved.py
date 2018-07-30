from __future__ import division, print_function
import numpy as np
import sys

if __name__ == '__main__':
    
    # Set parameters
    escale_factor = 1.0
    Lg = 2.0
    Rg = 0.15
    Nx = 12
    Ntheta = 5
    caps_layers = 1
    a = Rg * np.sin(np.pi / Ntheta)
    pattern = 'square' # square or hexagonal pattern

    # Scale Lg, Rg and a
    Lg = Lg * escale_factor
    Rg = Rg * escale_factor
    a  = a * escale_factor
    
    print('# a  = ', a)
    print('# Rg = ', Rg)
    print('# Lg = ', Lg)
    print('# caps_layers = ', caps_layers)

    # Calculate additional parameters
    dx = Lg / (Nx - 1)
    dtheta = 2 * np.pi / Ntheta
    dcaps = Rg / np.maximum(caps_layers, 1)
    
    # Print rod sides
    print(Nx * Ntheta)
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
            np.savetxt(sys.stdout, r[None, :], delimiter = '  ')


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
            np.savetxt(sys.stdout, r[None, :], delimiter = '  ')
            
        # Other cap
        rx = Lg * 0.5
        for itheta in range(Ntheta_layer):
            theta = dtheta_layer * (itheta + 0.5 * (ilayer % 2))
            ry = dcaps * ilayer * np.cos(theta)
            rz = dcaps * ilayer * np.sin(theta)
            
            r = np.array([rx, ry, rz])
            np.savetxt(sys.stdout, r[None, :], delimiter = '  ')
            
