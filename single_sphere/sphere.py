'''This module contains functions needed for generating heights according to the boltzmann distribution
for a sphere next to a wall'''

import numpy as np
import matplotlib.pyplot as plt

# parameters. Units are um, s, mg.
A = 0.265*np.sqrt(3./2.)   # Radius of blobs in um
ETA = 8.9e-4  # Water. Pa s = kg/(m s) = mg/(um s)

# density of particle = 0.2 g/cm^3 = 0.0000000002 mg/um^3
# volume is ~1.1781 um^3
weight = 1.1781*0.0000000002*(9.8*1.e6)
KT = 300.*1.3806488e-5  # T = 300K

# these were made up somewhat arbitrarily
REPULSION_STRENGTH = 7.5*KT
DEBYE_LENGTH = 0.5*A


# return the boltzmann distribution for one sphere
# the potential energy need only be composed of the gravitational potential
# energy and the energy from the wall repulsion
def single_sphere_generate_boltzmann_distribution(location, new_location):
	U = 0
	h = new_location[2]
	U += weight * h
	U += (REPULSION_STRENGTH * np.exp(-1. * h - A) / DEBYE_LENGTH) / (h - A)
	return np.exp(-1 * U / KT)


# this function performs the rejection algorithm for a single sphere.
# a height within the chosen bounds is randomly generated and an acceptance
# probability is formed using single_sphere_generate_boltzmann_distribution.
# a random value between 0 and 1 is compared to the acceptance probability
# and so a reasonable height is returned
def single_sphere_generate_equilibrium_sample_rejection(location, partitionZ):
	# formula for max_height chosen by Stephen. Works well enough, though I'm not sure why
	max_height = KT/weight*12 + A + 4.*DEBYE_LENGTH

	# generate heights and their corresponding probabilities until a height passes unrejected
	while True:
		new_location = [0., 0., np.random.uniform(A, max_height-A)]
		acceptance_prob = (single_sphere_generate_boltzmann_distribution(location,new_location))/partitionZ
		
		if np.random.uniform(0., 1.) < acceptance_prob: # the rejection part of the algorithm. 
			return new_location


# by sampling over 10,000 distributions, this function generates a reasonable normalization
# constant to use in single_sphere_generate_equilibrium_sample_rejection
# the function uses the maximum sample for the partition function and multiplies by the 
# constant 2 in order to ensure that the acceptance probability is below 1, always
def generate_partition_function(location):
	max_height = KT/weight*12 + A + 4.*DEBYE_LENGTH
	partitionZ = 0
	#for i in range(100):
	#	new_location = [0., 0., np.random.uniform(A, max_height-A)]
	#	partitionZ += single_sphere_generate_boltzmann_distribution(location,new_location)
	for i in range(10000):
		new_location = [0., 0., np.random.uniform(A, max_height-A)]
		sample = single_sphere_generate_boltzmann_distribution(location,new_location)
		if sample > partitionZ:
			partitionZ = sample
	return partitionZ*2


# generate the histogram of the heights by reading in the heights from the given file to x
def plot_distribution(locationsFile):
	x = np.loadtxt(locationsFile, float)
	plt.hist(x, 500, normed=1, facecolor='green', alpha=0.75)
	plt.title('Probability distribution of the height z of a single sphere near a wall')
	plt.xlabel('z (microns)')
	plt.ylabel('P(z)')
	plt.axis([0, 12.5, 0, .35])
	plt.show()
