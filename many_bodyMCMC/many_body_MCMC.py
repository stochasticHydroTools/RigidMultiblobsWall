import numpy as np
import time
import sys
import many_body_potential_pycuda as pycuda
sys.path.append('..')
from single_non_sphere import non_sphere
from body.body import Body
from quaternion_integrator.quaternion import Quaternion
from read_input import read_vertex_file, read_clones_file
from read_input.read_input import ReadInput
import utils

if len(sys.argv) != 2: # script takes input file containing .vertex and .clones files as command line argument
	print "Usage: python many_body_MCMC.py <input_file.txt>"
	sys.exit()
input_file = sys.argv[1]
movie_file = 'movies/movie_trajectories.txt' # file storing trajectories used in steven's animation code
trajectory_location, trajectory_orientation = [], [] # trajectory lists for steven's animation code
crosspoint_distances = []
angles_list = []
A = 0.265*np.sqrt(3./2.)
DIAM = 2*A
Lx = 0. # size of periodic boundary for x
Ly = 0.
max_translation = 1.
max_angle_shift = 0.1
VISCOSITY = 8.9e-4
WEIGHT = 1.*0.0000000002*(9.8*1e6)/7. # weight of blob
KT = 300.*1.3806488e-5
REPULSION_STRENGTH = 7.5 * KT
DEBYE_LENGTH_WALL = 0.5*A
DEBYE_LENGTH_PART = 0.5*A
max_starting_height = KT/(WEIGHT*7)*12 + A + 4.*DEBYE_LENGTH_WALL
epsilon = 0.095713728509
boom1_cross, boom2_cross = 6, 21 # for two size 15 boomerangs

input_blobs = ReadInput(input_file) # ReadInput object will parse input file
n_steps = input_blobs.n_steps

num_structures = len(input_blobs.structures) # different types of bodies with their own .vetex and .clones files
body = [] # list to contain each body used in simulation

# initialize each body to be used in the simulation using .vertex and .clone files
for i in range(num_structures):
	body_vectors = read_vertex_file.read_vertex_file( input_blobs.structures[i][0] )
	numBodies, body_locations, body_orientations = read_clones_file.read_clones_file( input_blobs.structures[i][1] )
	#print body_locations
	#print body_orientations
	for j in range(numBodies):
		body.append( Body(body_locations[j], body_orientations[j], body_vectors, A) )

# before MCMC begins, give all bodies random locations and orientations
# and initialize master array of r_vectors for all blobs in the simulation
sample_r_vectors = []
for i in range(len(body)):
	# body length used to prevent placing particles beyond boundaries to start with
	body_length = body[i].calc_body_length()
	if Lx != 0. and Ly != 0.: # if PBC active
		new_location = [np.random.uniform(body_length, Lx-body_length), np.random.uniform(body_length, Ly-body_length), np.random.uniform(body_length, max_starting_height)]
	else:
		new_location = [0,0, np.random.uniform(A, max_starting_height)]
	body[i].location = new_location
	body[i].orientation.random_orientation()
	sample_r_vectors.extend(body[i].get_r_vectors()) # add body's blob position vectors to sample list
sample_r_vectors = np.array(sample_r_vectors)

# begin MCMC
# get energy of the current state before jumping into the loop
start_time = time.time()
current_state_energy = pycuda.many_body_potential(sample_r_vectors,\
                                                  Lx, Ly,\
                                                  DEBYE_LENGTH_WALL,\
                                                  REPULSION_STRENGTH,\
                                                  DEBYE_LENGTH_PART,\
                                                  WEIGHT, KT,\
                                                  epsilon,\
                                                  boom1_cross,\
                                                  boom2_cross,\
                                                  DIAM, A)
quaternion_shift = Quaternion(np.array([1,0,0,0])) # quaternion to be used for disturbing the orientation of each body
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
		#print sample_r_vectors
		#print body[i].location[2]
		blob_index += body[i].Nblobs
	# calculate potential of proposed new state
	sample_state_energy = pycuda.many_body_potential(sample_r_vectors,\
                                                  Lx, Ly,\
                                                  DEBYE_LENGTH_WALL,\
                                                  REPULSION_STRENGTH,\
                                                  DEBYE_LENGTH_PART,\
                                                  WEIGHT, KT,\
                                                  epsilon,\
                                                  boom1_cross,\
                                                  boom2_cross,\
                                                  DIAM, A)
	# accept or reject the sample state and collect data accordingly
	if np.random.uniform(0.,1.) < np.exp(-(sample_state_energy - current_state_energy) / KT):
		current_state_energy = sample_state_energy
		#print current_state_energy
		# gather locations and orientations for movie made by plot_boomerang_trajectories.py
		for i in range(numBodies):
			body[i].location, body[i].orientation = body[i].location_new, body[i].orientation_new
	
	# collect data
	for i in range(numBodies):
		trajectory_location.append(body[i].location)
		trajectory_orientation.append(np.concatenate((np.array([body[i].orientation.s]), body[i].orientation.p)))

	crosspoint_distances.append(np.linalg.norm(body[0].get_r_vectors()[7]-body[1].get_r_vectors()[7]))

	#rint body[0].orientation.rotation_matrix()
	vec = np.matrix(((0,0,1)))
	angles_list.append(np.linalg.norm(vec * body[0].orientation.rotation_matrix()))
	angles_list.append(np.linalg.norm(vec * body[1].orientation.rotation_matrix()))
	#print body[0].get_r_vectors()[6]
	#print body[1].get_r_vectors()[6]
	#print current_state_energy
	#if current_state_energy == 0.:
	#	break

# function requires dictionary to be sent for additional parameters
# but we have no parameters we want to send, hence the empty dict
# steven's functions are a little convoluted
utils.write_trajectory_to_txt(movie_file, [trajectory_location,trajectory_orientation], {})

end_time = time.time() - start_time
print end_time

histogram_z_file = "hist_heights.txt"
# make histogram of collected heights
with open(histogram_z_file, 'w') as f:
	for boom_height in trajectory_location:
		f.write(str(boom_height[2]) + '\n')
non_sphere.plot_distribution(histogram_z_file, [], [], n_steps, 'green')

histogram_r_file = "hist_r.txt"
# make histogram of collected r vectors between crosspoints
with open(histogram_r_file, 'w') as f:
	for r in crosspoint_distances:
		f.write(str(r) + '\n')
non_sphere.plot_distribution(histogram_r_file, [], [], n_steps, 'green')

histogram_angle_file = "hist_angle.txt"
# make histogram of collected angles
with open(histogram_angle_file, 'w') as f:
	for angle in angles_list:
		f.write(str(angle) + '\n')
non_sphere.plot_distribution(histogram_angle_file, [], [], n_steps, 'green')