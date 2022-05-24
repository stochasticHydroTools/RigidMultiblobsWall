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
        from fiber import fiber
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

def set_mobility_blobs(implementation):
  '''
  Set the function to compute the dense mobility
  at the blob level to the right implementation.
  The implementation in C++ is somewhat faster than 
  the python one; to use it the user should compile 
  the file mobility/mobility.cpp

  These functions return an array with shape 
  (3*Nblobs, 3*Nblobs).
  '''
  # Implementations without wall
  if implementation == 'python_no_wall':
    return mb.rotne_prager_tensor
  if implementation == 'C++_no_wall':
    return mb.rotne_prager_tensor_cpp
  # Implementations with wall
  elif implementation == 'python':
    return mb.single_wall_fluid_mobility
  elif implementation == 'C++':
    return mb.single_wall_fluid_mobility_cpp
  # Implementation free surface
  elif implementation == 'C++_free_surface':
    return  mb.boosted_free_surface_mobility


def Mobility_Mult(X_force, r_vecs, eta, a, L=None):
    if L is None:
        L = np.array([0,0,0])
        
    #U = mb.single_wall_mobility_trans_times_force_pycuda(r_vecs, X_force, eta, a, periodic_length=L)
    U = mb.no_wall_mobility_trans_times_force_numba(r_vecs, X_force, eta, a, periodic_length=L)
    return U

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


def blob_blob_force(r, *args, **kwargs):
    '''
    This function compute the force between two blobs
    with vector between blob centers r.

    In this example the force is derived from the potential

    U(r) = U0 + U0 * (2*a-r)/b   if z<2*a
    U(r) = U0 * exp(-(r-2*a)/b)  iz z>=2*a

    with
    eps = potential strength
    r_norm = distance between blobs
    b = Debye length
    a = blob_radius
    '''
    # Get parameters from arguments
    dt = kwargs.get('time_step')
    eta = kwargs.get('eta')
    a = kwargs.get('blob_radius')
    eps_soft = kwargs.get('repulsion_strength') 
    b_soft = kwargs.get('debye_length')
    L = kwargs.get('periodic_length')
    # Compute force
    project_to_periodic_image(r, L)
    r_norm = np.linalg.norm(r)
    

    # Add wall interaction
    if r_norm > (2.0*a):
      force_torque = -((eps_soft / b_soft) * np.exp(-(r_norm-(2.0*a)) / b_soft) / np.maximum(r_norm, np.finfo(float).eps)) * r 
    else:
      force_torque = -((eps_soft / b_soft) / np.maximum(r_norm, np.finfo(float).eps)) * r;
    #r_hat = (1.0 / np.maximum(r_norm, np.finfo(float).eps)) * r
    #if r_norm > (2.0*a):
      #force_torque = 0.0 * r 
    #else:
      #Uprime = (16.0*np.pi*eta*(a**2)/dt)*(1.0-(2.0*a/r_norm))
      #force_torque = Uprime * r_hat;

    return force_torque

def calc_blob_blob_forces(r_vectors, *args, **kwargs):
    '''
    This function computes the blob-blob forces and returns
    an array with shape (Nblobs, 3).
    '''
    Nblobs = int(r_vectors.size / 3)
    force_blobs = np.zeros((Nblobs, 3))

    a = kwargs.get('blob_radius')
    L = kwargs.get('periodic_length')
    cut_pot = 3.5*a #3.5*a 
    put_r_vecs_in_periodic_box(r_vectors, L)
    r_tree = spatial.cKDTree(r_vectors,boxsize=L)

    for j in range(Nblobs):
        s1 = r_vectors[j]
        idx = r_tree.query_ball_point(s1,r=cut_pot) #2.2*a
        idx_trim = [i for i in idx if i > j]
        for k in idx_trim:
            if abs(k-j) == 1:
                continue
            s2 = r_vectors[k]
            r = s2-s1
            force = blob_blob_force(r, *args, **kwargs)
            force_blobs[j] += force
            force_blobs[k] -= force
    return force_blobs

if __name__ == '__main__':
    
        ### Set some physical parameters
        eta = 1.0 # viscosity
        L = 2.0 # fiber length
        
        ### Make an angled fiber
        ### For a *clamped* fiber, this serves as the 'ghost tangent' used to compute the bending force
        ### Note: if 'Clamp' is Flase this is passed by not used in the cpp code
        ### Note: it can be updated in the time loop to prescribe some mtion at the base of the fiber (e.g rotated in plane)
        theta = 0.0
        phi = np.pi/2.0-np.pi/8.0
        T_end = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
        
        
        Nlinks = 50
        ### Make array of tangent vectors and bishop frame
        tangents = np.tile(T_end, (Nlinks, 1))
        
        
        u,v = get_bishop(tangents)
        
        
        ds = L/(1.0*Nlinks)
        a = ds/2.0
        
        
        
        
        
        ### Make an Nfib_x-by-Nfib_y array of fibers at z=0 in a 4x4x4 periodic box
        Nfib_x = 10 # Number of fibers in x-direction
        dx_fib = 4.0/Nfib_x
        Nfib_y = 10 # Number of fibers in y-direction
        dy_fib = 4.0/Nfib_y
        
        ### Set the starting position of the first fiber
        X_0 = np.array([0.5*dx_fib,0.5*dy_fib,0.0])
        
        
        ### Make big matricees of size (3*Nfibers x Nlinks) for all of the fiber tangent vectors/bishop frame
        ### In this example, all of the fibers have the same tangents, but differnt x_0 to arrange them in a grid
        All_Taus = tangents.T
        All_X = X_0
        All_U = u.T
        All_V = v.T
        for kx in range(Nfib_x):
            for ky in range(Nfib_y):
                if kx==0 and ky == 0:
                    continue
                All_Taus = np.vstack((All_Taus, tangents.T))
                X_1 = X_0 + np.array([kx*dx_fib,ky*dy_fib,0.0])
                All_X = np.r_[All_X, X_1]
                All_U = np.vstack((All_U, u.T))
                All_V = np.vstack((All_V, v.T))
        
        ### Print the shapes of the fibers
        print(All_Taus.shape)
        print(All_X.shape)
        print(All_U.shape)
        print(All_V.shape)
        ### Print the toal number of fibers
        Nfibs = int(len(All_X)/3)
        print('Fibers: ' + str(Nfibs))
        
        ### Set the thermal energy and compute the bending stiffness to have 
        ### A prescribed persistance length
        kBT = 0.004
        #######################################
        k_b = 4.0*L*kBT 
        ######################################
        
        
        ### Physical scale for the mobility
        M0 = 1.0/(6.0*np.pi*eta*a) #np.log(L/a)/(8.0*np.pi*eta*a)
        ### Bending timescale
        alpha =((ds**3)/M0)/k_b
        ### dimensionless timestep
        alpha_fact = (2.0**4) ##################### Change this to change the time step
        ### Physcial time step
        dt = alpha_fact*alpha
        ### Factor for implicit time stepping
        impl = 0.5*dt*k_b/(ds*ds*ds)
        
        
        ### Print timestep and size of fiber
        print('dt = ' + str(dt))
        Nlk = np.shape(All_U)[1]
        Nblobs = Nlk+1
        print('Links: ' + str(Nlk))
        
        
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
        Clamp = True
        ###########################################
        ###########################################
        
        
        ###########################################
        #### Angular Velocity for twirling     ####
        ###########################################
        ###########################################
        Omega_base = 2.0*np.pi/(400.0*dt)
        ###########################################
        ###########################################
        
        ###########################################
        ########## Periodic Length   ##############
        ###########################################
        ###########################################
        Lperiodic = 4.0
        ###########################################
        ###########################################
        
        
        cf = cfibers.CManyFibers()
        cf.setParameters(a, ds, dt, k_b, M0, impl, kBT, eta, Lperiodic, Clamp, T_end)
        pos = cf.multi_fiber_Pos(All_Taus,All_X)
        
        plot_n = 20
        Nstep = int(400*plot_n)
        for k in range(Nstep):
            
          theta = Omega_base*k*dt
          phi = np.pi/2.0-np.pi/8.0
          T_end = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
          cf.update_T_fix(T_end)
          
          ### print config data to file at regular intervals
          if (k % plot_n == 0): # and (k > 0)
            pos = cf.multi_fiber_Pos(All_Taus,All_X)
            pos = np.reshape(pos,(3*Nfibs,Nblobs))
            print('saving config')
            np.savetxt('./Clamp_fiber_Data/Obj_Test_Det_Twirl_fibers_100x100_N100_data'+str(int(k/plot_n))+'.txt', pos)
            print(k)
            
          
          # Calculate steric forces between particles
          # Note: this is what we do for particles and it doesn't seem to work that well for fibers (but it's something *shrug*)
          start = time.time()
          pos = cf.multi_fiber_Pos(All_Taus,All_X)
          r_vecs = np.reshape(pos,(Nblobs*Nfibs,3))
          start = time.time()
          Forces = calc_blob_blob_forces(r_vecs, blob_radius=a, repulsion_strength=(16*kBT), debye_length=(0.1*a), periodic_length=np.array([3.9999, 3.9999, 3.9999]), eta=eta, time_step=dt )
          Forces = Forces.flatten()
          end = time.time()
          print("Time Force: "+str(end - start)+" s")
          
          
          # Setup RHS and find midpoint
          # Computes a midpoint configuration (in stochasitc case) and RHS for the solve
          RHS, T_h, U_h, V_h, X_h = cf.RHS_and_Midpoint(Forces, All_Taus, All_U, All_V, All_X)
          RHS_norm = np.linalg.norm(RHS);
          end = time.time()
          print("Time RHS: "+str(end - start)+" s")
          
          
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
          print("Num its: "+str(len(res_list)))
          print("Time Solve: "+str(end - start)+" s")
          
          # Evolve the fibers through rotation
          cf.frame_rot(All_Taus, All_U, All_V, All_X, Om, dt)
        end = time.time()
        print("Time "+str(end - start)+" s")
        
