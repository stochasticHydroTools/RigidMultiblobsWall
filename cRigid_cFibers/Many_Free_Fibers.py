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
import scipy.sparse as sp
from sksparse.cholmod import cholesky
import pyamg
import matplotlib.pyplot as plt
import c_fibers_obj as cfibers

# Find project functions
found_functions = False
path_to_append = ''
sys.path.append('../')
while found_functions is False:
    try:
        #from fiber import fiber
        from stochastic_forcing import stochastic_forcing as stochastic
        from mobility import mobility as mb
        from read_input import read_input
        from read_input import read_vertex_file
        from read_input import read_clones_file
        from read_input import read_slip_file
        import general_application_utils

        found_functions = True
    except ImportError:
        path_to_append += '../'
        print
        'searching functions in path ', path_to_append
        sys.path.append(path_to_append)
        if len(path_to_append) > 21:
            print
            '\nProjected functions not found. Edit path in multi_bodies.py'
            sys.exit()




def get_bishop(tangent, u_0 = None):
    '''
    Return the coordinates of the blobs.
    '''
    # Get location and orientation
    if u_0 is None:
        t_0 = tangent[0,:]
        u_0 = np.array([1.0,0.0,0.0])
        u_0 = u_0 - np.dot(u_0,t_0)*t_0
        u_0 /= np.linalg.norm(u_0)

    # Compute blobs coordinates
    u = 0*tangent 
    v = 0*tangent 

    u[0,:] = u_0
    v[0,:] = np.cross(tangent[0,:],u_0)

    for k in range((tangent.size // 3)-1):
        t_k = tangent[k,:]
        t_kp = tangent[k+1,:]
        cos_th = np.dot(t_k,t_kp)
        rot_x = np.cross(t_k,t_kp)
        
        u_k = u[k,:]
        v_k = v[k,:]
        
        u[k+1,:] = u_k + np.cross(rot_x,u_k) + (1.0/(1.0 + cos_th)) * np.cross(rot_x,np.cross(rot_x,u_k))
        v[k+1,:] = np.cross(t_kp,u[k+1,:])


    return u,v 

def project_to_periodic_image(r, L):
    '''
    Project a vector r to the minimal image representation
    centered around (0,0,0) and of size L=(Lx, Ly, Lz). If 
    any dimension of L is equal or smaller than zero the 
    box is assumed to be infinite in that direction.
    '''
    if L is not None:
        for i in range(3):
          if(L[i] > 0):
            r[i] = r[i] - int(r[i] / L[i] + 0.5 * (int(r[i]>0) - int(r[i]<0))) * L[i]
    return r

def put_r_vecs_in_periodic_box(r_vecs, L):
    for r_vec in r_vecs:
        for i in range(3):
          if L[i] > 0:
            while r_vec[i] < 0:
              r_vec[i] += L[i]
            while r_vec[i] > L[i]:
              r_vec[i] -= L[i]



if __name__ == '__main__':
    
        ### Set some physical parameters
        eta = 1.0 # viscosity
        L = 2.0 # fiber length
        
        ### Make an angled fiber
        ### For a *clamped* fiber, this serves as the 'ghost tangent' used to compute the bending force
        ### Note1: if 'Clamp' is Flase this is passed but not used in the cpp code
        ### Note2: it can be updated in the time loop to prescribe some motion at the base of the fiber (e.g rotated in plane)
        T_base = np.array([0.0,1.0,0.0])
        
        ###############################################################################################################
        # I made Nlinks the primary control variable, but it's pretty straightforward to change things around
        Nlinks = 10
        ### Make array of tangent vectors and bishop frame
        tangents = np.tile(T_base, (Nlinks, 1))
        u,v = get_bishop(tangents)
        # set a,ds based on Nlinks
        ds = L/(1.0*Nlinks)
        a = ds/2.0
        ###############################################################################################################
        
        
        ### Make an Nfib x 1 array of fibers at z=0,5a,10a,...
        Nfibs = 10
        X_0 = np.array([0.0,0.0,0.0])

        ### Make big matricees of size (3*Nfibers x Nlinks) for all of the fiber tangent vectors/bishop frame
        ### In this example, all of the fibers have the same tangents, but differnt x_0 to arrange them in a grid
        All_Taus = tangents.T
        All_X = X_0
        All_U = u.T
        All_V = v.T
        for kx in range(1,Nfibs):
            All_Taus = np.vstack((All_Taus, tangents.T))
            X_1 = X_0 + np.array([0.0,0.0,kx*5.0*a])
            All_X = np.r_[All_X, X_1]
            All_U = np.vstack((All_U, u.T))
            All_V = np.vstack((All_V, v.T))
        
        ### Print the shapes of the fibers
        print(All_Taus)
        print(All_X)
        print(All_U.shape)
        print(All_V.shape)
        ### Print the toal number of fibers
        Nfibs = int(len(All_X)/3)
        print('Fibers: ' + str(Nfibs))
      
        
        ### Set the thermal energy and compute the bending stiffness to have 
        ### A prescribed persistance length
        kBT = 0.004142
        #######################################
        k_b = 4.0*L*kBT # bending stiffness
        ######################################
        
        
        ### Physical scale for the mobility
        M0 = 1.0/(6.0*np.pi*eta*a) #np.log(L/a)/(8.0*np.pi*eta*a)
        ### Bending timescale
        alpha =((ds**3)/M0)/k_b
        ### dimensionless timestep
        alpha_fact = 1.0 ##################### Change this to change the time step
        ### Physcial time step
        dt = alpha_fact*alpha
        ### Factor for implicit time stepping
        ########################################################################################
        ########################################################################################
        ########################################################################################
        # Use impl_c=0.5 for CN
        # Use impl_c=1.0 for BE
        impl_c = 0.5
        impl = impl_c*dt*k_b/(ds*ds*ds)
        ########################################################################################
        ########################################################################################
        ########################################################################################
        
        
        ### Print timestep and size of fiber
        print('dt = ' + str(dt))
        Nlk = np.shape(All_U)[1]
        Nblobs = Nlk+1
        print('Blobs: ' + str(Nblobs))
        
        
        ###########################################
        #### Sets the number of neighbor blobs ####
        #### to be used in the PC              ####
        ###########################################
        M_bands = 0 # NOTE: 0 is fastest in most cases
        ###########################################
        ###########################################

        
        ###########################################
        #### Set kBT to zero to do a           ####
        #### determinsitic simulation          ####
        ###########################################
        ###########################################
        #kBT = 0.0 # NOTE: kBT is used to define bending stiffness ealier which is why we set it to zero here
        ###########################################
        ###########################################
        
        ###########################################
        #### Set Clamp to 'True' for a fiber   ####
        #### Bound to it's initial position at ####
        #### One of it's ends.                 ####
        #### Set Clamp to 'False' for a fiber  ####
        #### That is free at both ends         ####
        ###########################################
        ###########################################
        Clamp = False
        ###########################################
        ###########################################
        
        ###########################################
        ########## Periodic Length   ##############
        ###########################################
        ###########################################
        Lperiodic = -1 # Currently not used unless PSE is specified for the mobility
        ###########################################
        ###########################################
        
        ###########################################
        # Domain int specifies the mobility product 
        # to be used in the simulation. Right now
        # the values are:
        # DomainInt=0: Batched RPY (1 fiber per batch)
        # DomainInt=1: Full RPY
        # DomainInt=2: Batched RPB (1 fiber per batch)
        # DomainInt=3: Full RPB
        DomainInt = 0
        ###########################################
        
        cf = cfibers.CManyFibers()
        cf.setParameters(DomainInt, Nfibs, Nblobs, a, ds, dt, k_b, M0, impl_c, kBT, eta, Lperiodic, Clamp, T_base)
        pos = cf.multi_fiber_Pos(All_Taus,All_X)
        
        ############################################
        # output parameters
        out_e2e = True # wheter to output the end to end distance
        plot_n = 1e4 # output plot data every ... time steps
        Nstep = int(2*plot_n) # total number of timesteps
        for k in range(Nstep):
          if k % int(Nstep/50) == 0:
            print(str(100.0*k/(1.0*Nstep))+'% done')  
            
          if out_e2e:
              if k == 0:
                  status = 'w'
              else:
                  status = 'a'
              e2e = cf.end_to_end_distance(All_Taus)
              with open('./Free_Fiber_Data/end_to_end_distances.txt', status) as outfile:
                for ee in e2e:
                    outfile.write(str(ee) + " ")
                outfile.write("\n")
          
          
          ### print config data to file at regular intervals
          if (k % plot_n == 0): # and (k > 0)
            pos = cf.multi_fiber_Pos(All_Taus,All_X)
            pos = np.reshape(pos,(Nblobs,3*Nfibs))
            #pos = np.reshape(pos,(-1,3))
            print('saving config')
            np.savetxt('./Free_Fiber_Data/test_data'+str(int(k/plot_n))+'.txt', pos)
            print(k)
            
          
          # Calculate forces
          Forces = 0*np.zeros(Nblobs*Nfibs*3)
          
          start = time.time()
          # Setup RHS and find midpoint
          # Computes a midpoint configuration (in stochasitc case) and RHS for the solve
          RHS, T_h, U_h, V_h, X_h, BI = cf.RHS_and_Midpoint(Forces, All_Taus, All_U, All_V, All_X)
          RHS_norm = np.linalg.norm(RHS);
          end = time.time()
          #print("Time RHS: "+str(end - start)+" s")
          
          
          #define Right PC Linear Operator
          def A_x_PC(X):
            return cf.apply_A_x_Banded_PC(X, T_h, U_h, V_h, X_h, M_bands)
          Nsize = Nfibs*(3+3*Nlk)
          A = spla.LinearOperator((Nsize, Nsize), matvec=A_x_PC, dtype='float64')
          
          # Solve the linear system using GMRES
          res_list = [];
          start = time.time()
          
          (Sol, info_precond) = pyamg.krylov.gmres(A, (RHS/RHS_norm), x0=None, tol=1e-2, M=None,
                                                     maxiter=1, restrt=min(300, Nsize), residuals=res_list)
          
          # Extract the velocities from the GMRES solution
          Sol *= RHS_norm
          Om, Tens = cf.apply_Banded_PC(Sol, T_h, U_h, V_h, X_h, M_bands)
          end = time.time()
          #print("Num its: "+str(len(res_list)))
          #print("Time Solve: "+str(end - start)+" s")
          
          # Evolve the fibers through rotation
          cf.frame_rot(All_Taus, All_U, All_V, All_X, Om, dt)
        end = time.time()
        print("Time "+str(end - start)+" s")
        
