'''
Small test to verify that a uniformly distributed rotation
corresponds to a quaternion uniformly distributed on the surface
of the 3-Sphere.
'''
import sys
import numpy as np
import uniform_analyzer as ua
import cProfile, pstats, StringIO

PROFILE = 0

def generate_random_rotation_matrix():
  ''' 
  Generate a rotation matrix that represents a uniformly
  random distributed rotation.  This uses polar decomposition.
  '''
  A = np.matrix([np.random.normal(0., 1., 3) for _ in range(3)])
  U, sigma, V = np.linalg.svd(A)
  R = U*V.T
  return R

def generate_random_rotation_matrix_manual():
  ''' 
  Generate a random rotation matrix uniformly by choosing
  one axis at a time. 
  '''
  # Generate x, uniformly on the sphere.
  x = np.random.normal(0., 1., 3)
  x = x/np.linalg.norm(x)
  
  # Generate y, uniformly on the circle around x.
  p = np.random.normal(0., 1., 3)
  p = p/np.linalg.norm(p)
  y = np.cross(x, p)
  y = y/np.linalg.norm(y)
  
  # Generate z as x cross y.
  z = np.cross(x, y)
  z = z/np.linalg.norm(z)
  return np.matrix([x, y, z])

  
def MatrixToQuaternion(R):
  ''' 
  Convert rotation matrix to a quaternion. 
  The matrix R looks like (in terms of quaternion entries)

      [ p1^2 + s^2 - 1/2     p1p2 + sp3         p3p1 - sp2       ] 
  2 X [ p1p2 - sp3           p2^2 + s^2 - 1/2   p2p3 + sp1       ]
      [ p1p3 + sp2           p2p3 - sp1         p3^2 + s^2 - 1/2 ]

  One can then add and subtract entries to get rations of p_k/s, and
  then use the fact that the quaternion is unit norm to get the values.
  This code will generate the sign of s randomly, since the plus and minus 
  of a quaternion represent the same rotation.
  '''
  # Find ratios of p entries to s.
  p1_over_s = (R[2, 0] + R[0, 2])/(R[0, 1] - R[1, 0])
  p2_over_s = (R[1, 0] + R[0, 1])/(R[1, 2] - R[2, 1])
  p3_over_s = (R[1, 2] + R[2, 1])/(R[2, 0] - R[0, 2])
  
  # Find s_squared.
  s_squared = 1./(1. + p1_over_s**2 + p2_over_s**2 + p3_over_s**2)
  
  # Back out values of s and p.
  s = np.sqrt(s_squared)

  # + and - are the same rotation, pick a sign randomly 
  # to test against the uniform distribution.
  if np.random.randint(2) == 0:
    s = -1.*s
  p1 = p1_over_s*s
  p2 = p2_over_s*s
  p3 = p3_over_s*s
  
  return [s, p1, p2, p3]

  
if __name__ == "__main__":
  ''' Generate a number of samples, then test for uniformity '''
  if PROFILE:
    pr = cProfile.Profile()
    pr.enable()

  samples = []
  uniform_samples = []
  for k in range(int(sys.argv[1])):
    samples.append(MatrixToQuaternion(generate_random_rotation_matrix_manual()))
    x = np.random.normal(0., 1., 4)
    x = x/np.linalg.norm(x)
    uniform_samples.append(x)
  
  # Analyze distribution on 3-sphere
  rotation_analyzer = ua.UniformAnalyzer(samples, 'Rotations')
  uniform_analyzer = ua.UniformAnalyzer(uniform_samples, 'Uniform')
  ua.compare_distributions([rotation_analyzer, uniform_analyzer])
      
  if PROFILE:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()  

  
