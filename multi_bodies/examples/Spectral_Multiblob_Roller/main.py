import argparse
import numpy as np
import scipy.linalg
import scipy.spatial as spatial
import scipy.sparse.linalg as spla
import subprocess
try:
  import pickle as cpickle
except:
  try:
    import cpickle
  except:
    import _pickle as cpickle
from functools import partial
import sys
import time
import copy
import scipy.sparse as sp
from sksparse.cholmod import cholesky
#import matplotlib.pyplot as plt

# Find project functions
found_functions = False
path_to_append = ''
while found_functions is False:
    try:
        from DPStokesTests.python_interface.common_interface_wrapper import FCMJoint as FCMJoint 
        from stochastic_forcing import stochastic_forcing as stochastic
        from mobility import mobility as mob
        from body import body
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
            print('\nProjected functions not found. Edit path in multi_bodies.py')
            sys.exit()
            
            
            
def Wall_Mobility_Mult_Parts(X, r_vecs, eta, a, L):
    '''
    Multiply a vector X  of forces and torques by the RPB mobility. X should be formatted according to [F_1 T_1 F_2 T_2 ...]^{T} 
    '''
    num_particles = r_vecs.shape[0]

    FT = X.reshape(num_particles, 6)
    F = FT[:, 0:3].flatten()
    T = FT[:, 3:6].flatten()

    start = time.time()
    Mtt_x_F = mob.single_wall_mobility_trans_times_force_pycuda(r_vecs, F, eta, a, periodic_length=L)
    end = time.time()
    print('time Mtt_x_F: '+ str((end - start)) + ' (s)')
    
    start = time.time()
    Mtr_x_T = mob.single_wall_mobility_trans_times_torque_pycuda(r_vecs, T, eta, a, periodic_length=L)
    end = time.time()
    print('time Mtr_x_T: '+ str((end - start)) + ' (s)')
    
    start = time.time()
    Mrt_x_F = mob.single_wall_mobility_rot_times_force_pycuda(r_vecs, F, eta, a, periodic_length=L)
    end = time.time()
    print('time Mrt_x_F: '+ str((end - start)) + ' (s)')
    
    start = time.time()
    Mrr_x_T = mob.single_wall_mobility_rot_times_torque_pycuda(r_vecs, T, eta, a, periodic_length=L)
    end = time.time()
    print('time Mrr_x_T: '+ str((end - start)) + ' (s)')

    return Mtt_x_F, Mtr_x_T, Mrt_x_F, Mrr_x_T, F, T





if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description='Run a multi-body simulation and save trajectory.')
    parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', help='name of the input file')
    parser.add_argument('--print-residual', action='store_true', help='print gmres and lanczos residuals')
    args = parser.parse_args()
    input_file = args.input_file

    # Read input file
    read = read_input.ReadInput(input_file)

    # Set some variables for the simulation
    eta = read.eta
    a = read.blob_radius
    
    ##############
    particle_radius = a
    ##############
    
    
    output_name = read.output_name
    structures = read.structures
    print(structures)
    structures_ID = read.structures_ID
    
    # Copy input file to output
    subprocess.call(["cp", input_file, output_name + '.inputfile'])

    # Create rigid bodies
    bodies = []
    body_types = []
    body_names = []
    for ID, structure in enumerate(structures):
        print('Creating structures = '+structure[1])
        # Read vertex and clones files
        struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
        
        ############################################
        struct_ref_config *= particle_radius
        print(struct_ref_config)
        import scipy.spatial.distance as distance 
        if len(struct_ref_config) > 1:
            dist = distance.pdist(struct_ref_config)
            a = 0.5*(dist.min())
        else:
            a = particle_radius
        print('\n')
        print('a is: '+str(a))
        ############################################
        
        num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])
        # Read slip file if it exists
        slip = None
        if(len(structure) > 2):
            slip = read_slip_file.read_slip_file(structure[2])
        body_types.append(num_bodies_struct)
        body_names.append(structures_ID[ID])
        # Create each body of type structure
        for i in range(num_bodies_struct):
            b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, a)
            b.ID = structures_ID[ID]
            # Calculate body length for the RFD
            if i == 0:
                b.calc_body_length()
            else:
                b.body_length = bodies[-1].body_length
            # Append bodies to total bodies list
            bodies.append(b)
    bodies = np.array(bodies)

    # Set some more variables
    num_of_body_types = len(body_types)
    num_bodies = bodies.size
    Nblobs = sum([x.Nblobs for x in bodies])
    cutoff = 4.5
    
    L = read.periodic_length
    
    n_steps = read.n_steps 
    n_save = read.n_save
    dt = read.dt 
    
    for b in bodies:
      for i in range(3):
        if L[i] > 0:
           while b.location[i] < 0:
             b.location[i] += L[i]
           while b.location[i] > L[i]:
             b.location[i] -= L[i]
                        
                        
    
    r_vecs_blobs = np.concatenate([b.get_r_vectors() for b in bodies])
    for vec in r_vecs_blobs:
      for i in range(3):
        if L[i] > 0:
           while vec[i] < 0:
             vec[i] += L[i]
           while vec[i] > L[i]:
             vec[i] -= L[i]
    
    
    nP = len(r_vecs_blobs)
    print('# of blobs: ' + str(nP))
    
    
    #################################################
    ########## Set Up Fast Multiply Object ##########
    #################################################
    device = 'gpu' 
    domType = 'DPBW' # for bottom wall
    has_torque = True #Set to True if angular displacements are needed 
    #Simulation domain limits
    xmin = 0.0; xmax = L[0]
    ymin = 0.0; ymax = L[1]
    zmin = 0.0; zmax = 9.1740639106166668
    # Initialize the solver with all the parameters
    solver = FCMJoint(device)
    #Initialize can be called several times in order to change the parameters
    solver.Initialize(numberParticles=Nblobs, hydrodynamicRadius=a, kernType=0,
                    domType=domType, has_torque=has_torque,
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                    viscosity=eta, optInd=0, ref=False)

    t0 = time.time()
    solver.SetPositions(r_vecs_blobs.flatten())
    dt1 = time.time() - t0
    print('set positions: ' + str(dt1))
    #################################################
    #################################################
    
    

    #print(data.shape)
    head='particle_rad: '+str(particle_radius)+'\tblob_rad: '+str(a)
    #np.savetxt('./data/Cfg_'+str(len(bodies[0].get_r_vectors()))+'_blobs_per.txt',r_vecs_blobs,fmt='%.9f',header=head)
    
    
    np.random.seed(0)
    t0 = time.time()
    X = np.random.normal(0.0, 1.0, 6*Nblobs)
    Mtt_x_F, Mtr_x_T, Mrt_x_F, Mrr_x_T, F, T = Wall_Mobility_Mult_Parts(X,r_vecs_blobs, eta, a, L)
    dt1 = time.time() - t0
    print('n^2 GPU time, M*[F, T]: ' + str(dt1))
    
    t0 = time.time()
    V, W = solver.Mdot(F, T)
    dt1 = time.time() - t0
    print('DPStokes time, M*[F, T]: ' + str(dt1))
    
    print('M*F test')
    t0 = time.time()
    VV = solver.Mdot_MB(r_vecs_blobs, F, eta, a)
    dt1 = time.time() - t0
    print('DPStokes time, M*[F]: ' + str(dt1))
    
    vel_error = np.linalg.norm(V - Mtt_x_F - Mtr_x_T)/np.linalg.norm(Mtt_x_F + Mtr_x_T)
    omega_error = np.linalg.norm(W - Mrt_x_F - Mrr_x_T)/np.linalg.norm(Mrt_x_F + Mrr_x_T)
    MF_error = np.linalg.norm(VV - Mtt_x_F)/np.linalg.norm(Mtt_x_F)
    
    
    print('Velocity Error: ' + str(vel_error))
    print('Omega Error: ' + str(omega_error))
    print('MF Error: ' + str(MF_error))
    
    data = np.column_stack((F, T, Mtt_x_F, Mtr_x_T, Mrt_x_F, Mrr_x_T))
    #np.savetxt('./data/Mob_Product_'+str(len(bodies[0].get_r_vectors()))+'_blobs_per.txt',data,fmt='%.9f')
      
      
    ########################################
    # GPU Results
    ########################################
    # for 1 blob
    # n^2 GPU time, M*[F, T]: 0.46822595596313477
    # DPStokes time, M*[F, T]: 0.03192496299743652

    # for 12 blobs, 
    # n^2 GPU time, M*[F, T]: 41.68016195297241
    # DPStokes time, M*[F, T]: 0.7058537006378174

    # for 42 blobs, gpu DPStokes runs out of memory

