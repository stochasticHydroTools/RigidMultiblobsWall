import argparse
import numpy as np
import scipy.linalg
import scipy.spatial as spatial
import scipy.sparse.linalg as spla
import subprocess
from shutil import copyfile
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
    import multi_bodies_functions
    from mobility import mobility as mb
    from quaternion_integrator.quaternion import Quaternion
    from quaternion_integrator.quaternion_integrator_multi_bodies import QuaternionIntegrator
    from quaternion_integrator.quaternion_integrator_rollers import QuaternionIntegratorRollers
    from body import body 
    from read_input import read_input
    from read_input import read_vertex_file
    from read_input import read_clones_file
    from read_input import read_slip_file
    import general_application_utils as utils        
    try:
      import libCallHydroGrid as cc
      found_HydroGrid = True
    except ImportError:
      found_HydroGrid = False
    found_functions = True
  except ImportError:
    path_to_append += '../'
    print('searching functions in path ', path_to_append)
    sys.path.append(path_to_append)
    if len(path_to_append) > 21:
      print('\nProjected functions not found. Edit path in multi_bodies.py')
      sys.exit()


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
    has_torque = False #Set to True if angular displacements are needed 
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
    
    print('solver successfully initialized')
             
    solver.SetPositions(r_vecs_blobs.flatten())
    np.random.seed(1)
    forces_test = np.random.normal(0.0, 1.0, 3*Nblobs)
    torques_test = np.random.normal(0.0, 1.0, 3*Nblobs)
    print('test F,T')
    V, W = solver.Mdot(forces_test,0*torques_test)
    print(V[0:3])
    print(W[0:3])
    print('test MB')
    MF = solver.Mdot_MB(r_vecs_blobs, forces_test, eta, a, periodic_length=L)
    
