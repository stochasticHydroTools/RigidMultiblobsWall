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

class InvalidProbability(Exception):
	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)


# return the boltzmann distribution for one sphere
# the potential energy need only be composed of the gravitational potential
# energy and the energy from the wall repulsion
def single_sphere_GB(new_location):
	U = 0
	h = new_location[2]

	U += weight * h
	U += (REPULSION_STRENGTH * np.exp(-1. * h - A) / DEBYE_LENGTH) / (h - A)
	return np.exp(-1 * U / KT)


# this function performs the rejection algorithm for a single sphere.
# a height within the chosen bounds is randomly generated and an acceptance
# probability is formed using single_sphere_GB.
# a random value between 0 and 1 is compared to the acceptance probability
# and so a reasonable height is returned
def single_sphere_rejection(partitionZ):
	# generate heights and their corresponding probabilities until a height passes unrejected
	while True:
		new_location = [0., 0., np.random.uniform(A, max_height-A)]
		acceptance_prob = (single_sphere_GB(new_location))/partitionZ
		if acceptance_prob > 1:
			raise InvalidProbability('Acceptance Probability is greater than 1')
		if np.random.uniform(0., 1.) < acceptance_prob: # the rejection part of the algorithm. 
			return new_location


# by generating 10,000 samples for the distribution, this function generates a normalization
# constant to use in single_sphere_rejection
# the function uses the maximum sample for the partition function and multiplies by the 
# constant 2 in order to ensure that the acceptance probability is below 1, always
def generate_partition(partition_steps):
	partitionZ = 0
	#for i in range(100):
	#	new_location = [0., 0., np.random.uniform(A, max_height)]
	#	partitionZ += single_sphere_GB(location,new_location)
	for i in range(partition_steps):
		new_location = [0., 0., np.random.uniform(A, max_height)]
		sample = single_sphere_GB(new_location)
		if sample > partitionZ:
			partitionZ = sample
	return partitionZ*2


# generate the histogram of the heights by reading in the heights from the given file to x
# and plot the analytical distribution curve given by x and y
# bar width h chosen to be approximately n_steps^(-1/5)
# so for 1,000,000 steps, 357 bars are used for max_height ~ 22.5 um
def plot_distribution(locationsFile, analytical_x, analytical_y, n_steps):
	heights = np.loadtxt(locationsFile, float)
	# the hist function returned a 3rd item and I'm not sure how best to handle it yet
	# so there is a throwaway variable trash
	numBars = int(max_height // (n_steps**(-1/5.)))
	binValue, xBinLocations, trash = plt.hist(heights, numBars, normed=1, facecolor='green', alpha=0.75)
	plt.plot(analytical_x, analytical_y, 'b-', linewidth=1.5)
	
	# add error bars to histogram	Nx = # samples in bin	h = bin width
	# N = total number of samples generated
	# error bar length = (4 * sqrt(Nx)) / (h*N)
	binWidth = xBinLocations[1] - xBinLocations[0]
	xError, yError, confidence = [], [], []
	for i in range(binValue.size):
		xError.append( (xBinLocations[i+1]+xBinLocations[i]) / 2) # center bin i
		yError.append(binValue[i]) # height of bin i
		numSamples = binWidth * binValue[i] * n_steps # samples in bin i
		confidence.append( (4 * np.sqrt(numSamples)) / (binWidth * n_steps)) 
	plt.errorbar(xError,yError,yerr=confidence,fmt='r.')

	plt.title('Probability distribution of the height z of a single sphere near a wall\n' + 
			  'Green: histogram of sampled heights  Blue: GB distribution')
	plt.xlabel('z (microns)')
	plt.ylabel('P(z)')
	plt.axis([0, 12.5, 0, .35])
	plt.show()


# calculate an num_points numbver of points given by directly computing the Gibbs-Boltzmann distribution
# P(h) = exp(-U(h)/KT) / integral(exp(U(h)/KT)dh)
# calculated using the trapezoidal rule
def analytical_distribution(num_points):
	# heights are sampled evenly from the chosen bounds, using linspace
	# because linspace includes starting value A, the first index in x is ignored
	# if x[0] is included, then in the calculation of potential energy U, h-A = 0
	# and an exception will be thrown
	x = np.linspace(A, max_height, num_points) 
	y = []
	deltaX = x[1] - x[0] 
	numerator, denominator = 0., 0.

	# add the bounds to the integral value
	# ignore x[0] = A
	integral = 0.5*(single_sphere_GB([0., 0., x[1]]) + 
					single_sphere_GB([0., 0., max_height]))
	# iterate over the rest of the heights
	for k in range(2,num_points):
		integral += single_sphere_GB([0., 0., x[k]])
	# multiply by the change in x to complete the integral calculation
	integral *= deltaX

	# now that we have the partition function that the integral represents
	# we can calculate all the y positions of the distribution 
	# again ignore x[0] = A
	for h in x[1:]:
		numerator = single_sphere_GB([0., 0., h])		
		y.append(numerator/integral)
	
	return x[1:], y
