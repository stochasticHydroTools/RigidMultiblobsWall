import numpy as np
import time
import sys
import many_body_potential_pycuda as pycuda
sys.path.append('..')
from body.body import Body
from quaternion_integrator.quaternion import Quaternion
from read_input import read_input
from read_input import read_vertex_file, read_clones_file
import utils


if __name__ == '__main__':

  # script takes input file as command line argument or default 'data.main'
  if len(sys.argv) != 2: 
    input_file = 'data.main'
  else:
    input_file = sys.argv[1]

  # Read input file
  read = read_input.ReadInput(input_file) 

  # file storing trajectories used in steven's animation code
  movie_file = read.output_name + '.trajectories.txt' 

  # trajectory lists for steven's animation code
  trajectory_location, trajectory_orientation = [], [] 
  blob_radius = read.blob_radius
  DIAM = 2 * blob_radius
  # size of periodic boundary for x
  Lx = 0. 
  Ly = 0.
  max_translation = 1.
  max_angle_shift = 0.1
  VISCOSITY = 8.9e-4
  WEIGHT = 1.*0.0000000002*(9.8*1e6)/7. # weight of blob
  KT = 300.*1.3806488e-5
  REPULSION_STRENGTH = 7.5 * KT
  DEBYE_LENGTH_WALL = 0.5 * blob_radius
  DEBYE_LENGTH_PART = 0.5 * blob_radius
  max_starting_height = KT/(WEIGHT*7)*12 + blob_radius + 4.*DEBYE_LENGTH_WALL
  epsilon = 0.095713728509
  boom1_cross, boom2_cross = 6, 21 # for two size 15 boomerangs
  

  n_steps = read.n_steps
  
  num_structures = len(read.structures) # different types of bodies with their own .vetex and .clones files
  body = [] # list to contain each body used in simulation
  
  # initialize each body to be used in the simulation using .vertex and .clone files
  for i in range(num_structures):
    body_vectors = read_vertex_file.read_vertex_file( read.structures[i][0] )
    numBodies, body_locations, body_orientations = read_clones_file.read_clones_file( read.structures[i][1] )
    for j in range(numBodies):
      body.append( Body(body_locations[j], body_orientations[j], body_vectors, blob_radius) )

  # before MCMC begins, give all bodies random locations and orientations
  # and initialize master array of r_vectors for all blobs in the simulation
  sample_r_vectors = []
  for i in range(len(body)):
    # body length used to prevent placing particles beyond boundaries to start with
    body_length = body[i].calc_body_length()
    if Lx != 0. and Ly != 0.: # if PBC active
      new_location = [np.random.uniform(body_length, Lx-body_length), np.random.uniform(body_length, Ly-body_length), np.random.uniform(body_length, max_starting_height)]
    else:
      new_location = [0,0, np.random.uniform(blob_radius, max_starting_height)]
    body[i].location = new_location
    body[i].orientation.random_orientation()
    sample_r_vectors.extend(body[i].get_r_vectors()) # add body's blob position vectors to sample list
  sample_r_vectors = np.array(sample_r_vectors)

  # begin MCMC
  # get energy of the current state before jumping into the loop
  start_time = time.time()
  current_state_energy = pycuda.many_body_potential(sample_r_vectors,
                                                    Lx, Ly,
                                                    DEBYE_LENGTH_WALL,
                                                    REPULSION_STRENGTH,
                                                    DEBYE_LENGTH_PART,
                                                    WEIGHT, KT,
                                                    epsilon,
                                                    boom1_cross,
                                                    boom2_cross,
                                                    DIAM, blob_radius)

  # quaternion to be used for disturbing the orientation of each body
  quaternion_shift = Quaternion(np.array([1,0,0,0]))

  # for each step in the Markov chain, disturb each body's location and orientation and obtain the new list of r_vectors
  # of each blob. Calculate the potential of the new state, and accept or reject it according to the Markov chain rules:
  # 1. if Ej < Ei, always accept the state  2. if Ej < Ei, accept the state according to the probability determined by
  # exp(-(Ej-Ei)/KT). Then record data.
  # Important: record data also when staying in the same state (i.e. when a sample state is rejected)
  for step in range(n_steps):
    blob_index = 0
    for i in range(numBodies): # distrub bodies
      body[i].location_new = body[i].location + np.random.uniform(-max_translation,max_translation, 3) # make small change to location
      quaternion_shift = Quaternion.from_rotation(np.random.normal(0,1,3) * max_angle_shift)
      body[i].orientation_new = quaternion_shift * body[i].orientation
      sample_r_vectors[blob_index : blob_index + body[i].Nblobs] = body[i].get_r_vectors(body[i].location_new, body[i].orientation_new)
      blob_index += body[i].Nblobs

    # calculate potential of proposed new state
    sample_state_energy = pycuda.many_body_potential(sample_r_vectors,
                                                     Lx, Ly,
                                                     DEBYE_LENGTH_WALL,
                                                     REPULSION_STRENGTH,
                                                     DEBYE_LENGTH_PART,
                                                     WEIGHT, KT,
                                                     epsilon,
                                                     boom1_cross,
                                                     boom2_cross,
                                                     DIAM, blob_radius)

    # accept or reject the sample state and collect data accordingly
    if np.random.uniform(0.,1.) < np.exp(-(sample_state_energy - current_state_energy) / KT):
      current_state_energy = sample_state_energy
      for i in range(numBodies):
        body[i].location, body[i].orientation = body[i].location_new, body[i].orientation_new
	
    # collect data
    for i in range(numBodies):
      trajectory_location.append(body[i].location)
      trajectory_orientation.append(np.concatenate((np.array([body[i].orientation.s]), body[i].orientation.p)))


  # function requires dictionary to be sent for additional parameters
  # but we have no parameters we want to send, hence the empty dict
  # steven's functions are a little convoluted
  utils.write_trajectory_to_txt(movie_file, [trajectory_location,trajectory_orientation], {})

  end_time = time.time() - start_time
  print end_time
