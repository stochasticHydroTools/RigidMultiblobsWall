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
# formula for max_height chosen by Stephen. Works well enough, though I'm not sure why
max_height = KT/weight*12 + A + 4.*DEBYE_LENGTH



# return the boltzmann distribution for one sphere
# the potential energy need only be composed of the gravitational potential
# energy and the energy from the wall repulsion
def single_sphere_generate_boltzmann_distribution(new_location):
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
def single_sphere_generate_equilibrium_sample_rejection(partitionZ):
	# generate heights and their corresponding probabilities until a height passes unrejected
	while True:
		new_location = [0., 0., np.random.uniform(A, max_height-A)]
		acceptance_prob = (single_sphere_generate_boltzmann_distribution(new_location))/partitionZ
		
		if np.random.uniform(0., 1.) < acceptance_prob: # the rejection part of the algorithm. 
			return new_location


# by sampling over 10,000 distributions, this function generates a reasonable normalization
# constant to use in single_sphere_generate_equilibrium_sample_rejection
# the function uses the maximum sample for the partition function and multiplies by the 
# constant 2 in order to ensure that the acceptance probability is below 1, always
def generate_partition_function():
	partitionZ = 0
	#for i in range(100):
	#	new_location = [0., 0., np.random.uniform(A, max_height-A)]
	#	partitionZ += single_sphere_generate_boltzmann_distribution(location,new_location)
	for i in range(10000):
		new_location = [0., 0., np.random.uniform(A, max_height-A)]
		sample = single_sphere_generate_boltzmann_distribution(new_location)
		if sample > partitionZ:
			partitionZ = sample
	return partitionZ*2


# generate the histogram of the heights by reading in the heights from the given file to x
# and plot the analytical distribution curve given by x and y
def plot_distribution(locationsFile, x, y):
	heights = np.loadtxt(locationsFile, float)
	plt.hist(heights, 500, normed=1, facecolor='green', alpha=0.75)
	plt.plot(x,y,'b-', linewidth=1.5)
	plt.title('Probability distribution of the height z of a single sphere near a wall\n' + 
			  'Green: histogram of sampled heights  Blue: GB distribution')
	plt.xlabel('z (microns)')
	plt.ylabel('P(z)')
	plt.axis([0, 12.5, 0, .35])
	plt.show()


# calculate 100,000 points given by directly solving the Gibbs-Boltzmann distribution
# P(h) = exp(-U(h)/KT) / integral(exp(U(h)/KT)dh)
# calculated using the trapezoidal rule
def analytical_distribution():
	# 100,000 heights are sampled evenly from the chosen bounds, 1.5 is somewhat arbitrary
	# 0 is not chosen because I was getting an overflow in potential energy U when the
	# sphere was very close to the wall
	x = np.linspace(1.5, max_height, 100000) 
	y = []
	deltaX = x[1] - x[0] 
	numerator, denominator = 0., 0.

	# add the bounds to the integral value
	integral = 0.5*(single_sphere_generate_boltzmann_distribution([0., 0., 1.5]) + 
					single_sphere_generate_boltzmann_distribution([0., 0., max_height]))
	# iterate over the rest of the heights
	for k in np.linspace(x[1], x[99998], 99998):
		integral += single_sphere_generate_boltzmann_distribution([0., 0., k])
	# multiply by the change in x to complete the integral calculation
	integral *= deltaX

	# now that we have the partition function that the integral represents
	# we can calculate all the y positions of the distribution 
	for h in x:
		numerator = single_sphere_generate_boltzmann_distribution([0., 0., h])		
		y.append(numerator/integral)
	
	return x, y