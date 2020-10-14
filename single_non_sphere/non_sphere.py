
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
	#for k in range(len(r_vectors)):
	#	if r_vectors[k][2] < A:
	#		return 0.0
	U = 0.
	for k in range(len(r_vectors)):
		if r_vectors[k][2] < A:
			return 0.0
		U += WEIGHT[k] * r_vectors[k][2]
		h = r_vectors[k][2]
		# Add repulsion to potential.
		U += ( REPULSION_STRENGTH * np.exp(-1.*(h -A)/DEBYE_LENGTH) / (h-A) )
	#U *= REPULSION_STRENGTH
	#U += np.sum(WEIGHT[0]*r_vectors[:,2])
	return np.exp(-1. * U/KT)


def get_boomerang_r_vectors(location, orientation, blob_distance = .525):
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
        
        Donev: Even if you hard code a value (which I recommend against -- 
        you will learn some new programming skills trying to figure out how to pass this as an argument.
        Ask Floren -- python supports optional arguments with default values. He has had to do something like this.
        But even if hard-coded, write
        const=0.525
        and then use const in the code. This way you can change it with one-line.
        It is a really really bad idea to hard-code values like this..
	'''
	num_blobs = 7
	initial_configuration = [  np.array([3*blob_distance, 0., 0.]),
							 			np.array([2*blob_distance, 0., 0.]),
						 	 			np.array([blob_distance, 0., 0.]),
						 	 			np.array([0., 0., 0.]),
						 	 			np.array([0., blob_distance, 0.]),
						 	 			np.array([0., 2*blob_distance, 0.]),
						 	 			np.array([0., 3*blob_distance, 0.])]

	rotation_matrix = orientation.rotation_matrix()
	rotated_configuration = [np.dot(rotation_matrix,vec) + location for vec in initial_configuration]
	#rotated_configuration = []
 	#for vec in initial_configuration:
	#	rotated_configuration.append(np.dot(rotation_matrix, vec)
	#							   + location)

	return rotated_configuration



def generate_non_sphere_partition(partition_steps):
	partitionZ = 0.
	#for i in range(100):
	#	new_location = [0., 0., np.random.uniform(A, max_height)]
	#	partitionZ += non_sphere_GB(location,new_location)
	orientation = Quaternion([0,0,0,0])
	new_location = np.array([0., 0., 0.])
	for i in range(partition_steps):
		orientation.random_orientation()
		new_location[2] = np.random.uniform(A, max_height)
		sample = non_sphere_GB(new_location, orientation)
		if sample > partitionZ:
			partitionZ = sample
	return partitionZ



def non_sphere_rejection(partitionZ):
	# generate heights and their corresponding probabilities until a height passes unrejected
	orientation = Quaternion([0.,0.,0.,0.])
	new_location = np.array([0., 0., 0.])
	while True:
		orientation.random_orientation()
		new_location[2] = np.random.uniform(A, max_height-A)
		acceptance_prob = non_sphere_GB(new_location, orientation) / partitionZ
		if acceptance_prob > 1:
			raise InvalidProbability('Acceptance Probability is greater than 1')
		if np.random.uniform(0., 1.) < acceptance_prob: # the rejection part of the algorithm. 
			return [new_location, orientation]



# calculate an num_points numbver of points given by directly computing the Gibbs-Boltzmann distribution
# P(h) = exp(-U(h)/KT) / integral(exp(U(h)/KT)dh)
# calculated using the trapezoidal rule
# Donev: Explain to me in person what this does
def analytical_distribution_non_sphere(num_points):
	# heights are sampled evenly from the chosen bounds, using linspace
	# because linspace includes starting value A, the first index in x is ignored
	# if x[0] is included, then in the calculation of potential energy U, h-A = 0
	# and an exception will be thrown
#	x = np.linspace(A, max_height, num_points)
#	orientations = [] 
#	for i in range(num_points):
#		theta = np.random.normal(0., 1., 4)
#		orientations.append(Quaternion(theta/np.linalg.norm(theta)))
#	y = []
#	deltaX = x[1] - x[0] 
#	numerator, denominator = 0., 0.
#
#	# add the bounds to the integral value
#	# ignore x[0] = A
#	integral = 0.5*(non_sphere_GB([0., 0., x[1]], orientations[0]) + 
#					non_sphere_GB([0., 0., max_height], orientations[num_points-1]))
#	# iterate over the rest of the heights
#	for k in range(2, num_points):
#		integral += non_sphere_GB([0., 0., x[k]], orientations[k])
#	# multiply by the change in x to complete the integral calculation
#	integral *= deltaX
#
#	# now that we have the partition function that the integral represents
#	# we can calculate all the y positions of the distribution 
#	# again ignore x[0] = A
#	j = 0
#	for h in x[1:]:
#		numerator = non_sphere_GB([0., 0., h], orientations[j])		
#		y.append(numerator/integral)
#		j+=1


	
	x = np.linspace(A, max_height, num_points)
	y = np.zeros(num_points-1, dtype = float)
	num_angles = 1000
	deltaX = x[1] - x[0]
	integral = .0
	firstBar, lastBar = 0., 0.
	orientation = Quaternion([0,0,0,0]) # create a quaternion object

	for i in range(num_angles):
		orientation.random_orientation()
		firstBar += non_sphere_GB([0, 0, x[1]], orientation)
	firstBar /= num_angles

	for i in range(num_angles):
		orientation.random_orientation()
		lastBar += non_sphere_GB([0, 0, x[num_points-1]], orientation)
	lastBar /= num_angles

	integral += (firstBar + lastBar) *.5

	sample_GB = np.zeros(num_angles, dtype = float)
	for i in range(2, num_points-1):
		for j in range(num_angles):
			orientation.random_orientation()
			sample_GB[j] = non_sphere_GB(np.array([0, 0, x[i]]), orientation)
		integral += np.average(sample_GB)
	integral *= deltaX

	for i in range(x[1:].size):
		numerator = 0.
		for j in range(num_angles):
			orientation.random_orientation()
			numerator += non_sphere_GB([0., 0., x[i+1]], orientation)
		numerator /= num_angles
		y[i] = (numerator/integral)


	return x[1:], y




# generate the histogram of the heights by reading in the heights from the given file to x
# and plot the analytical distribution curve given by x and y
# bar width h chosen to be approximately n_steps^(-1/5)
# so for 1,000,000 steps, 357 bars are used for max_height ~ 22.5 um
def plot_distribution(locationsFile, analytical_x, analytical_y, n_steps, color):
	heights = np.loadtxt(locationsFile, float)
	# the hist function returned a 3rd item and I'm not sure how best to handle it yet
	# so there is a throwaway variable trash
	numBars = int(max_height // (n_steps**(-1/5.)))
	binValue, xBinLocations, trash = plt.hist(heights, numBars, normed=1, facecolor=color, alpha=0.75)
	plt.hist(heights, numBars, normed=1, facecolor=color, alpha=0.75)
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
	#plt.errorbar(xError,yError,yerr=confidence,fmt='r.')

#	p, q = [], []
#	skip = 1000
#	size = 100000
#	for i in range(0, size-skip, skip):
#		average = 0.
#		for j in range(i,i+skip):
#			average += analytical_y[j]
#			#print j
#		average /= float(skip)
#		#print("%f   %f" % (analytical_x[i], average))
#		p.append(analytical_x[i])
#		q.append(average)

	plt.plot(analytical_x, analytical_y, 'b.-', linewidth=1.5)
	#plt.plot(p, q, 'b-', linewidth=2)

	plt.title('Probability distribution of the height z of a single boomerang near a wall\n' + 
			  'Green: histogram of sampled heights  Blue: GB distribution')
	plt.xlabel('z (microns)')
	plt.ylabel('P(z)')
	plt.axis([0, 12.5, 0, .35])
	plt.show()
