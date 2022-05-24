import argparse
import numpy as np
import scipy.linalg as la
import scipy.spatial as spatial
import scipy.sparse.linalg as spla
import subprocess
from functools import partial
import sys
import time
import copy
import subprocess
import scipy.sparse as sp
from sksparse.cholmod import cholesky
import pyamg
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as pyrot

# Find project functions
found_functions = False
path_to_append = './'
sys.path.append('../')
sys.path.append('../DoublyPeriodicStokes/python_interface/.')


for i in range(10):
    path_to_append += '../'
    sys.path.append(path_to_append)

import c_rigid_obj as cbodies
from read_input import read_input
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_slip_file
import general_application_utils


def intialize_DPStokes(zmin,zmax,Lx,Ly,radP,domType,kernTypes=2,optInd=0):
    # use kernType 0,1 for single blob 
    has_torque = 0
    ref = False
    Lx, Ly, hx, hy, nx, ny, w, w_d, cbeta, cbeta_d, beta, beta_d\
        = interface.configure_grid_and_kernels_xy(Lx, Ly, np.ones((1,))*radP, np.ones((1,))*kernTypes, optInd, has_torque, ref)
    zmax, hz, nz, zmin = interface.configure_grid_and_kernels_z(zmin, zmax, hx, w, w_d, domType, fac=1.5, ref=ref)
    if domType == 'TP':
        mode = 'periodic'
    elif domType == 'DP':
        mode = 'nowall'
    elif domType == 'DPBW':
        mode = 'bottom'
    elif domType == 'DPSC':
        mode = 'slit'
    
    cb.setParametersDP(nx, ny, nz, Lx, Ly, zmin, zmax, w[0], w_d[0], w[0]*beta[0], w_d[0]*beta_d[0], mode)


if __name__ == '__main__':
        # Get command line arguments
        parser = argparse.ArgumentParser(description='Run a multi-body simulation and save trajectory.')
        parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', help='name of the input file')
        parser.add_argument('--print-residual', action='store_true', help='print gmres and lanczos residuals')
        args = parser.parse_args()
        input_file = args.input_file

        # Read input file
        read = read_input.ReadInput(input_file)
        
        
        import os
        (exDir, trash) = os.path.split(input_file)
        sys.path.append(exDir)
        print(sys.path)
        import multi_bodies_functions
        #import common_interface_wrapper as interface

        # Set some variables for the simulation
        a = read.blob_radius
        output_name = exDir+read.output_name
        structures = read.structures
        print(structures)
        structures_ID = read.structures_ID
        

        
        # Copy input file to output
        subprocess.call(["cp", input_file, output_name + '.inputfile'])

        # Create rigid bodies
        X_0 = []
        Quat = []
        for ID, structure in enumerate(structures):
            if(ID > 0):
                print('Multiple structures not supported with cBodies yet :( Try using multi_bodeis.py')
                sys.exit()
            print('Creating structures = ', structure[1])
            # Read vertex and clones files
            Cfg = read_vertex_file.read_vertex_file(structure[0])
            num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(
                exDir+structure[1])
            # Read slip file if it exists
            slip = None
            if(len(structure) > 2):
                slip = read_slip_file.read_slip_file(exDir+structure[2])
            # Create each body of type structure
            for i in range(num_bodies_struct):
                X_0.append(struct_locations[i])
                Quat.append(struct_orientations[i].entries)


        Nbods = len(X_0)
        X_0 = np.array(X_0).flatten()
        Quat = np.array(Quat).flatten()

        import scipy.spatial.distance as distance 
        if len(Cfg) > 1 and (a <= 0):
            dist = distance.pdist(Cfg)
            a = 0.5*(dist.min())
        else:
            print('sigle blobs not supported with cBodies :( Try using lubrication')
            sys.exit()
           
        
           
           
        # read in misc. parameters
        n_steps = read.n_steps 
        n_save = read.n_save
        eta = read.eta
        dt = read.dt
        g = read.g
        kBT = read.kT
        Tol = read.solver_tolerance
        periodic_length = np.array(read.periodic_length)
        
        #read geomoetry parameters
        domType=read.domType
        zmin=read.zmin
        if(zmin != 0):
            print('non-zero zmin not supported :(')
            sys.exit()
        zmax=read.zmax
        
        
        # set repulsive potential based on 'a'
        repulsion_strength_wall = 4*kBT
        debye_length_wall = 0.1*a
        repulsion_strength = 4*kBT
        debye_length = 0.1*a
        
        
        #Example specific parameters
        Lcyl=2.0
        
        
        print('a is: '+str(a))
        print('diffusive blob timestep is: '+str(kBT*dt/(6*np.pi*eta*a**3)))
    
        
        # Make solver object
        cb = cbodies.CManyBodies()
        
        # Sets the PC type
        # If true will use the block diag 'Dilute suspension approximation' to M
        # For dense suspensions this is a bad approximation (try setting false)
        # for rigid bodies with lots of blobs this is expensive (try setting flase)
        cb.setBlkPC(True)
        
        # set Domain parameters
        Lx = periodic_length[0]
        Ly = periodic_length[1]

        print(domType)


        if domType == 'RPY':
            DomainInt=0
        elif domType == 'RPB':
            DomainInt=1
        elif domType[0] == 'D':
            intialize_DPStokes(zmin,zmax,Lx,Ly,a,domType)
            DomainInt = 2
        elif domType == 'TP':
            DomainInt = 3
        else:
            print('Domain type not supported') 
            sys.exit()
        numParts = Nbods*len(Cfg)
        cb.setParameters(numParts, a, dt, kBT, eta, periodic_length, Cfg, DomainInt)
        cb.setConfig(X_0,Quat)
        print('set config')
        cb.set_K_mats()
        print('set K mats')
        
        
        ######################################################################################################
        ########################################## Solver params ###############################################
        ######################################################################################################
        
        sz = 3*Nbods*len(Cfg)
        Nsize = sz + 6*Nbods
        
        # set the type of PC based on the domain
        if (domType == 'DPSC') or (domType == 'DPBW') or (domType == 'RPB'):
            cb.setWallPC(True)
        else:
            cb.setWallPC(False)
            
        Force = np.zeros(6*Nbods)
 
        
        num_rejects = 0
        Sol = np.zeros(Nsize)
        
        for n in range(n_steps):
            
            Qs, Xs = cb.getConfig()
            r_vectors = np.array(cb.multi_body_pos())
            FT = multi_bodies_functions.force_torque_calculator_all_bodies(Xs, Qs, r_vectors, g = g, blob_radius = a,
                                               z_max = zmax, Lrod = Lcyl, eta = eta, domType=domType,
                                               repulsion_strength_wall = repulsion_strength_wall, 
                                               debye_length_wall = debye_length_wall, 
                                               repulsion_strength = repulsion_strength, 
                                               debye_length = debye_length, 
                                               periodic_length = periodic_length)
            

            Force = FT.flatten()
            Slip = np.zeros(sz)
                
            
            start = time.time()
            RHS,VarKill,VarKill_strat = cb.RHS_and_Midpoint(Slip, Force)
            
            RHS_norm = np.linalg.norm(RHS)
            end = time.time()
            print("Time RHS: "+str(end - start)+" s")
            
            r_vectors_mid = np.array(cb.multi_body_pos())
            r_vectors_mid = np.reshape(r_vectors_mid, (Nbods*len(Cfg), 3))
            
            A = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_Saddle, dtype='float64')
            PC = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_PC, dtype='float64')
            
            res_list = [];
            start = time.time()
            
            (Sol, info_precond) = pyamg.krylov.fgmres(A, (RHS/RHS_norm), x0=None, tol=Tol, M=PC, #
                                                        maxiter=min(300, Nsize), restrt=None, residuals=res_list)
            
            #print(info_precond)
            
            # Extract the velocities from the GMRES solution
            #print(res_list)
            
            Sol *= RHS_norm
            Lambda_s = Sol[0:sz]
            U_s = Sol[sz::]
            
            end = time.time()
            
            cb.evolve_X_Q(U_s)
            
            print("Time Solve: "+str(end - start)+" s")
            print("Num its: " + str(len(res_list)))
 
            r_vectors_new = np.array(cb.multi_body_pos())
            r_vectors_new = np.reshape(r_vectors_new, (Nbods*len(Cfg), 3))
            print('num rejected: '+str(num_rejects))
            if(min(r_vectors_mid[:,2]) < a or max(r_vectors_mid[:,2]) > zmax-a):
                num_rejects += 1
                ########################
                print('Bad Timestep!!')
                cb.setConfig(Xs,Qs)
                continue
            
            if n >= 0 and (n % n_save == 0):
                name = output_name + '.config'
                if n == 0:
                    status = 'w'
                else:
                    status = 'a'
                with open(name, status) as f_ID:
                    f_ID.write(str(Nbods) + '\n')
                    for j in range(Nbods):
                        f_ID.write('%s %s %s %s %s %s %s\n' % (Xs[3*j+0],
                                                               Xs[3*j+1],
                                                               Xs[3*j+2],
                                                               Qs[4*j+0],
                                                               Qs[4*j+1],
                                                               Qs[4*j+2],
                                                               Qs[4*j+3]))


       
