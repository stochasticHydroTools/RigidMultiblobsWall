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
from os import path
sys.path.append('../')
import c_fibers_obj as cfibers

# Find project functions
found_functions = False
path_to_append = ''
while found_functions is False:
    try:
        import c_fibers_obj as cfibers

        found_functions = True
    except ImportError:
        path_to_append += "../"
        print('searching functions in path ', path_to_append)
        print(sys.path)
        sys.path.append(path_to_append)
        if len(path_to_append) > 21:
            print('\nProjected functions not found. Edit path in interacting_fibers.py')
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


def blob_blob_force(r, *args, **kwargs):
    '''
    This function computes the force between two blobs
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
      force_torque = -((eps_soft / b_soft) / np.maximum(r_norm, np.finfo(float).eps)) * r
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
   
   
    cf = cfibers.CManyFibers()

    for two_pow in [5]:
        ## Big fiber suspension
        Nblobs = 100
        L = 2.0
        ds = L/(Nblobs-1)
        a = ds/2.0
       
       
        print(int(4/a))
        UpsampledLocs = np.loadtxt("./cfg_800.txt")
       
        All_Taus = []
        All_X = []
        All_U = []
        All_V = []
       
        refine = 1
        Nstop = 10000
        kount = 0
        for fib in UpsampledLocs:
            if kount == Nstop:
                break
            kount += 1
            fib_rs = np.reshape(fib,(Nblobs,3))
            X_0 = fib_rs[0,:]
            All_X.append(X_0)
            tan = []
            for k in range(0,Nblobs-refine,refine):
                r = fib_rs[k+refine,:] - fib_rs[k,:]
                r = r/np.linalg.norm(r)
                tan.append(r)
            tangents = np.stack(tan, axis=0)
           
            u,v = get_bishop(tangents)
            All_Taus.append(tangents.T)
            All_U.append(u.T)
            All_V.append(v.T)
       
        Nfibs = len(All_X)
        print("Nfibers: ", Nfibs)
       
        All_X = np.concatenate(All_X, axis=0)
        All_Taus = np.concatenate(All_Taus, axis=0)
        All_U = np.concatenate(All_U, axis=0)
        All_V = np.concatenate(All_V, axis=0)
        print(All_Taus.shape)
        print(All_X.shape)
        print(All_U.shape)
        print(All_V.shape)

       
        ds *= refine
        a *= refine
        Nblobs /= refine
        Nblobs = int(Nblobs)

        kBT = 0.004
        #######################################
        k_b = 4.0*L*kBT #USED TO BE 2
        eta = 0.1
        ######################################
       
        M0 = np.log(L/a)/(8.0*np.pi*eta*a)
       
        alpha =((ds**3)/M0)/k_b
        alpha_fact = 50 #0.5*(2.0**two_pow)
        dt = alpha_fact*alpha
       
        print('dt = ' + str(dt))
        Nlk = np.shape(All_U)[1]
        print('Links: ' + str(Nlk))
        impl = 0.5*dt*k_b/(ds*ds*ds)
        M_bands = 0

        double=np.float64
        single=np.float32

        precision = double

        DomainInt = 0
        k_bend = L*kBT
        impl_c = 1.0
        Clamp = False
        Lperiodic = -1
        T_base = np.array([0.0,1.0,0.0], dtype=precision) # not used in c code if Clamp is False


        cf.setParameters(DomainInt, Nfibs, Nblobs, a, ds, dt, k_bend, M0, impl_c, kBT, eta, Lperiodic, Clamp, T_base)
       
       
        #################################
       
        Nstep = 400
        print('Number of steps: '+str(Nstep))
       
        S_time = [] 
        #################################
       
       
        output_positions = True
        plot_n = 20
        save_n = 200

        for k in range(Nstep):
            print(k)

            if output_positions and k % plot_n == 0:
                print("SAVING FIBER POS")
                if k == 0:
                    status = 'w'
                else:
                    status = 'a'

                pos = cf.multi_fiber_Pos(All_Taus,All_X)
                pos = np.asarray(pos)
                with open("./multi_fiber_data/fiber_pos.txt", status) as outfile:
                    pos.tofile(outfile, " ")
                    outfile.write("\n")
                    # exit()

            # if (k % plot_n == 0): # and (k > 0)
            #     pos = cf.multi_fiber_Pos(All_Taus,All_X)
            #     pos = np.reshape(pos,(3*Nfibs,Nblobs))
            #     print('saving config')
            #     np.savetxt('./multi_fiber_data/Shear_data'+str(int(k/plot_n))+'.txt', pos)
            #     print(k)
            
            # if ((k % save_n == 0) or (k == Nstep-1)): # and (k > 0)
            if ((k == Nstep-1)): # and (k > 0)
                print('saving data')
                np.savetxt('./multi_fiber_data/Shear_All_Taus_'+str(int(k))+'.txt', All_Taus)
                np.savetxt('./multi_fiber_data/Shear_All_U_'+str(int(k))+'.txt', All_U)
                np.savetxt('./multi_fiber_data/Shear_All_V_'+str(int(k))+'.txt', All_V)
                np.savetxt('./multi_fiber_data/Shear_All_X_'+str(int(k))+'.txt', All_X)
            
            # Setup RHS and find midpoint
            start = time.time()
            pos = cf.multi_fiber_Pos(All_Taus,All_X)
            r_vecs = np.reshape(pos,(Nblobs*Nfibs,3))
            #print(r_vecs)
            
            start = time.time()
            Forces = calc_blob_blob_forces(r_vecs, blob_radius=a, repulsion_strength=(16*kBT), debye_length=(0.1*a), periodic_length=np.array([3.9999, 3.9999, 3.9999]), eta=eta, time_step=dt )
            Forces = Forces.flatten()

            RHS, T_h, U_h, V_h, X_h, BI = cf.RHS_and_Midpoint(Forces, All_Taus, All_U, All_V, All_X)
            RHS_norm = np.linalg.norm(RHS)

            def A_x_PC(X):
                return cf.apply_A_x_Banded_PC(X, T_h, U_h, V_h, X_h, M_bands)
        
            Nsize = Nfibs*(3+3*Nlk)
            A = spla.LinearOperator((Nsize, Nsize), matvec=A_x_PC, dtype=precision)

            # Solve the linear system using GMRES
            res_list = []

            (Sol, info_precond) = pyamg.krylov.gmres(A, (RHS/RHS_norm), x0=None, tol=1e-4, M=None,
                                                        maxiter=1, restrt=min(300, Nsize), residuals=res_list)
            

            # Extract the velocities from the GMRES solution
            Sol *= RHS_norm
            Om, Tens = cf.apply_Banded_PC(Sol, T_h, U_h, V_h, X_h, M_bands)

            # Evolve the fibers through rotation
            cf.frame_rot(All_Taus, All_U, All_V, All_X, Om, dt)
       