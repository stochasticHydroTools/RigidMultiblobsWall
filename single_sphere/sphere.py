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
# Donev: Make sure to discuss with Blaise why this may or may not work:
# One needs to at the GB distribution tail and figure out where it decays to below, say, 10/N
# where N is the total number of (successful) MC trials. Also think about why I said 10/N and how one would justify this
# This requires you to understand how to compute error bars on the histogram
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
# and so a reasonable height is returned # Donev: What does "reasonable" mean?
# Donev: Consider shortening the names of your subroutines/functions/methods
# Donev: The fact that this code requires as input Z is bad.
# Estimating Z by a monte Carlo integral as you do is low accuracy and so this whole thing becomes low-accuracy
# regardless of how many samples you have generated. This is not good -- it means the error is not "controlled"
# "controlled-accuracy" is the single most important thing about numerical methods
# How can we improve on this? Let me think a bit more. Maybe Markov Chain MC is the only way to go...
def single_sphere_generate_equilibrium_sample_rejection(partitionZ):
	# generate heights and their corresponding probabilities until a height passes unrejected
	while True:
		new_location = [0., 0., np.random.uniform(A, max_height-A)]
		acceptance_prob = (single_sphere_generate_boltzmann_distribution(new_location))/partitionZ
		
		if np.random.uniform(0., 1.) < acceptance_prob: # the rejection part of the algorithm. 
			return new_location


# by sampling over 10,000 distributions, # Donev: this is 10K *samples*, there is only one distribution
# Pay attention to terminology since this that way you will understand the math better
# this function generates a reasonable normalization
# constant to use in single_sphere_generate_equilibrium_sample_rejection
# the function uses the maximum sample for the partition function and multiplies by the 
# constant 2 in order to ensure that the acceptance probability is below 1, always
# Donev: Never hard-wire values in code (we discussed this)
# Instead of 10,000, make this an *input* argument to this routine so you can improve accuracy if needed
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
# Donev: Add error bars to the histogram
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


# calculate 100,000 points given by directly computing the Gibbs-Boltzmann distribution
# P(h) = exp(-U(h)/KT) / integral(exp(U(h)/KT)dh)
# calculated using the trapezoidal rule
# Donev: This routine again has hard-wired values and smells of "hacking"
# When you tell me "I was getting an overflow in potential energy U" it means you should spend time
# to understand what the issue is and how to fix it properly instead of hacking it. This is how you learn
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
