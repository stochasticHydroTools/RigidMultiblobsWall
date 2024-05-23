# import argparse
import numpy as np
# import scipy.linalg as la
# import scipy.spatial as spatial
import scipy.sparse.linalg as spla
# import subprocess
# from functools import partial
import sys
import time
# import copy
# import scipy.sparse as sp
# from sksparse.cholmod import cholesky
import pyamg
# import matplotlib.pyplot as plt

# import os
# bin_dir = os.path.dirname(__file__) + "/bin"
sys.path.insert(0, "../") # needs to contain the .so file from compilation
import RigidFibers

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
        print('searching functions in path ', path_to_append)
        sys.path.append(path_to_append)
        if len(path_to_append) > 21:
            print
            '\nProjected functions not found. Edit path in Many_Free_Fibers.py'
            sys.exit()




def get_bishop(tangent, u_0 = None):
    '''
    Return the coordinates of the blobs.
    '''
    # Get location and orientation
    if u_0 is None:
        t_0 = tangent[0,:]
        u_0 = np.array([1.0,0.0,0.0], dtype=rf.precision)
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

    # exit()
    print("RUNNING")

    rf = RigidFibers.RigidFibers("./input_files/ManyFree.in", __file__)
    cf = rf.cf

    rf.make_run_directory()

    ### Make array of tangent vectors and bishop frame
    T_base = np.array([0.0,1.0,0.0], dtype=rf.precision)
    N_links = rf.N_blobs-1
    tangents = np.tile(T_base, (N_links, 1))
    u,v = get_bishop(tangents)

    ### Make an Nfib x 1 array of fibers at z=0,5a,10a,...
    Nfibs = rf.N_fibers
    X_0 = np.array([0.0,0.0,0.0], dtype=rf.precision)

    ### Make big matrices of size (3*Nfibers x Nlinks) for all of the fiber tangent vectors/bishop frame
    ### In this example, all of the fibers have the same tangents, but differnt x_0 to arrange them in a grid
    All_Taus = tangents.T
    All_X = X_0
    All_U = u.T
    All_V = v.T
    for kx in range(1,Nfibs):
        All_Taus = np.vstack((All_Taus, tangents.T))
        X_1 = X_0 + np.array([0.0,0.0,kx*5.0*rf.a], dtype=rf.precision)
        All_X = np.r_[All_X, X_1]
        All_U = np.vstack((All_U, u.T))
        All_V = np.vstack((All_V, v.T))
    
    
    ###########################################
    #### Sets the number of neighbor blobs ####
    #### to be used in the preconditioner  ####
    ###########################################
    M_bands = 0 # NOTE: 0 is fastest in most cases
    ###########################################
    ###########################################

    All_Taus = All_Taus.astype(rf.precision)
    All_X = All_X.astype(rf.precision)

    pos = cf.multi_fiber_Pos(All_Taus,All_X)
    pos = np.reshape(pos, (Nfibs, N_links+1, 3))

    start = time.time()

    a_hat = 1.1204*rf.a
    eps_hat = a_hat/rf.L_f
    t_bar = 0.0008*(4*np.pi*pow(rf.L_f, 4)*rf.eta)/(rf.k_bend*np.log(1/eps_hat))
    Nstep = round(t_bar/rf.dt)
    print(((rf.ds**3)/rf.M0)/rf.k_bend)
    print("dt: ", rf.dt)
    print("stop time: ", t_bar)
    print("Nsteps: ", Nstep)
    print(rf.M0)

    print("normalized persistence length (should be 1): ", (rf.k_bend/rf.kBT)/rf.L_f)
    print("eps hat (should be 0.01)", a_hat/rf.L_f)
    print(np.shape(All_Taus))
    
    iter_start = time.time()
    sim_start = time.time()
    for k in range(Nstep):
        if k % int(Nstep/50) == 0:
            print(str(100.0*k/(1.0*Nstep))+'% done')

        # iter_end = time.time()
        # print("itr time:", iter_end-iter_start)
        # iter_start = time.time()


        if rf.save_e2e:
            rf.save_e2e_dist(All_Taus)


        # Calculate forces
        Forces = 0*np.zeros(rf.N_blobs*rf.N_fibers*3, dtype=rf.precision)

        # Setup RHS and find midpoint
        # Computes a midpoint configuration (in stochasitc case) and RHS for the solve
        RHS, T_h, U_h, V_h, X_h, BI = cf.RHS_and_Midpoint(Forces, All_Taus, All_U, All_V, All_X)
        RHS_norm = np.linalg.norm(RHS)

        #define Right PC Linear Operator
        def A_x_PC(X):
            return cf.apply_A_x_Banded_PC(X, T_h, U_h, V_h, X_h, M_bands)
        
        Nsize = Nfibs*(3+3*N_links)
        A = spla.LinearOperator((Nsize, Nsize), matvec=A_x_PC, dtype=rf.precision)

        # Solve the linear system using GMRES
        res_list = []

        (Sol, info_precond) = pyamg.krylov.gmres(A, (RHS/RHS_norm), x0=None, tol=1e-4, M=None,
                                                    maxiter=1, restrt=min(300, Nsize), residuals=res_list)

        # Extract the velocities from the GMRES solution
        Sol *= RHS_norm
        Om, Tens = cf.apply_Banded_PC(Sol, T_h, U_h, V_h, X_h, M_bands)

        # Evolve the fibers through rotation
        cf.frame_rot(All_Taus, All_U, All_V, All_X, Om, rf.dt)

    sim_end = time.time()
    print("Time "+str(sim_end - sim_start)+" s")


        
