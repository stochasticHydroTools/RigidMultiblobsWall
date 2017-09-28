import numpy as np
import sys

if __name__ == '__main__':
    
    # Set parameters
    Lg = 1.70508734
    Rg = 0.155400002
    Nx = 24
    Ntheta = 12
    caps_layers = 2

    # Calculate additional parameters
    dx = Lg / (Nx - 1)
    dtheta = 2 * np.pi / Ntheta
    dcaps = Rg / np.maximum(caps_layers, 1)
    
    # Print rod sides
    print Nx * Ntheta 
    for ix in range(Nx):
        for itheta in range(Ntheta):
            rx = ix * dx - Lg * 0.5

            theta = dtheta * (itheta + 0.5 * (ix % 2))
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
            
