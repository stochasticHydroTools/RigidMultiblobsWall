import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from quaternion_integrator.quaternion import Quaternion

num_blobs = 7
A = 0.265*np.sqrt(3./2.) # radius of blob in um
VISCOSITY = 8.9e-4
TOTAL_WEIGHT = 1.*0.0000000002*(9.8*1e6) # weight of entire boomerang particle
WEIGHT = [TOTAL_WEIGHT/num_blobs for i in range(num_blobs)] # weight of individual blobs
KT = 300.*1.3806488e-5
REPULSION_STRENGTH = 7.5 * KT
DEBYE_LENGTH = 0.5*A
max_height = KT/TOTAL_WEIGHT*12 + A + 4.*DEBYE_LENGTH


class InvalidProbability(Exception):
	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)


def non_sphere_GB(location, orientation):
	''' Return exp(-U/kT) for the given location and orientation.'''
	r_vectors = get_boomerang_r_vectors(location, orientation)
	# Add gravity to potential.
	for k in range(len(r_vectors)):
		if r_vectors[k][2] < A:
			return 0.0
	U = 0
	for k in range(len(r_vectors)):
		U += WEIGHT[k]*r_vectors[k][2]
		h = r_vectors[k][2]
		# Add repulsion to potential.
		U += ( REPULSION_STRENGTH * np.exp(-1.*(h -A)/DEBYE_LENGTH) / (h-A) )

	return np.exp(-1. * U/KT)


def get_boomerang_r_vectors(location, orientation):
	'''Get the vectors of the 7 blobs used to discretize the boomerang.
  
		 7 O  
		 6 O  
		 5 O
		   O-O-O-O
		   4 3 2 1
	
	The location is the location of the Blob at the apex.
	Initial configuration is in the
	x-y plane, with  arm 1-2-3  pointing in the positive x direction, and arm
	4-5-6 pointing in the positive y direction.
	Seperation between blobs is currently hard coded at 0.525 um
	'''
	
	initial_configuration = [np.array([1.575, 0., 0.]),
							 np.array([1.05, 0., 0.]),
						 	 np.array([0.525, 0., 0.]),
						 	 np.array([0., 0., 0.]),
						 	 np.array([0., 0.525, 0.]),
						 	 np.array([0., 1.05, 0.]),
						 	 np.array([0., 1.575, 0.])]

	rotation_matrix = orientation.rotation_matrix()
	rotated_configuration = []
 	for vec in initial_configuration:
		rotated_configuration.append(np.dot(rotation_matrix, vec)
								 + np.array(location))

	return rotated_configuration



def generate_non_sphere_partition(partition_steps):
	partitionZ = 0.
	#for i in range(100):
	#	new_location = [0., 0., np.random.uniform(A, max_height)]
	#	partitionZ += non_sphere_GB(location,new_location)
	for i in range(partition_steps):
		theta = np.random.normal(0., 1., 4)
		orientation = Quaternion(theta/np.linalg.norm(theta))
		new_location = [0., 0., np.random.uniform(A, max_height)]
		sample = non_sphere_GB(new_location, orientation)
		if sample > partitionZ:
			partitionZ = sample
	return partitionZ*1.1



def non_sphere_rejection(partitionZ):
		# generate heights and their corresponding probabilities until a height passes unrejected
	while True:
		theta = np.random.normal(0., 1., 4)
		orientation = Quaternion(theta/np.linalg.norm(theta))
		new_location = [0., 0., np.random.uniform(A, max_height-A)]
		acceptance_prob = non_sphere_GB(new_location, orientation) / partitionZ
		if acceptance_prob > 1:
			raise InvalidProbability('Acceptance Probability is greater than 1')
		if np.random.uniform(0., 1.) < acceptance_prob: # the rejection part of the algorithm. 
			return [new_location, orientation]



# calculate an num_points numbver of points given by directly computing the Gibbs-Boltzmann distribution
# P(h) = exp(-U(h)/KT) / integral(exp(U(h)/KT)dh)
# calculated using the trapezoidal rule
def analytical_distribution_non_sphere(num_points):
	# heights are sampled evenly from the chosen bounds, using linspace
	# because linspace includes starting value A, the first index in x is ignored
	# if x[0] is included, then in the calculation of potential energy U, h-A = 0
	# and an exception will be thrown
	x = np.linspace(A, max_height, num_points)
	orientations = [] 
	for i in range(num_points):
		theta = np.random.normal(0., 1., 4)
		orientations.append(Quaternion(theta/np.linalg.norm(theta)))
	y = []
	deltaX = x[1] - x[0] 
	numerator, denominator = 0., 0.

	# add the bounds to the integral value
	# ignore x[0] = A
	integral = 0.5*(non_sphere_GB([0., 0., x[1]], orientations[0]) + 
					non_sphere_GB([0., 0., max_height], orientations[num_points-1]))
	# iterate over the rest of the heights
	for k in range(2,num_points):
		integral += non_sphere_GB([0., 0., x[k]], orientations[k])
	# multiply by the change in x to complete the integral calculation
	integral *= deltaX

	# now that we have the partition function that the integral represents
	# we can calculate all the y positions of the distribution 
	# again ignore x[0] = A
	j = 0
	for h in x[1:]:
		numerator = non_sphere_GB([0., 0., h], orientations[j])		
		y.append(numerator/integral)
		j+=1
	return x[1:], y



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

	p, q = [], []
	skip = 100
	size = 100000
	for i in range(0, size-skip, skip):
		average = 0.
		for j in range(i,i+skip):
			average += analytical_y[j]
			#print j
		average /= float(skip)
		#print("%f   %f" % (analytical_x[i], average))
		p.append(analytical_x[i])
		q.append(average)

	#plt.plot(analytical_x, analytical_y, 'bo', linewidth=1.5)
	plt.plot(p, q, 'b-', linewidth=2)

	plt.title('Probability distribution of the height z of a single boomerang near a wall\n' + 
			  'Green: histogram of sampled heights  Blue: GB distribution')
	plt.xlabel('z (microns)')
	plt.ylabel('P(z)')
	plt.axis([0, 12.5, 0, .35])
	plt.show()