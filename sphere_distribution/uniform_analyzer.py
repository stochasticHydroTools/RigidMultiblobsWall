'''
Class to look at samples on a sphere and verify that they are a uniform
distribution.
'''
import numpy as np

class UniformAnalyzer(object):
  ''' 
  This object just takes samples on a sphere and analyzes their 
  distribution to determine if they are uniformly distributed.
  '''
  def __init__(self, samples):
    ''' 
    Just copy the reference to the list of samples.
    Each sample should be of the same length, and represent
    a point on the sphere.
    '''
    self.samples = samples
    self.dim = len(self.samples[0])

  def AnalyzeSamples(self):
    ''' Analyze samples by calculating means of spherical harmonics. '''
    for k in range(5):
      print '-'*60
      print 'xi, eta pair: ', k
      xi, eta = self.GenerateXiEta()
    
      for L in range(1, 11):
        harmonics = []
        for sample in self.samples:
          u = np.inner(xi, sample)
          v = np.inner(eta, sample)
          theta = np.arctan(v/u)
          harmonics.append(cos(l*theta))
        print "Mean at L = %d is: %d" % (L, mean(harmonics))


  def GenerateXiEta(self):
    ''' Generate a random pair of orthonormal vectors. '''
    xi = np.random.normal(0, 1, self.dim)
    xi = xi/np.linalg.norm(xi)
    
    eta = np.random.normal(0, 1, self.dim)
    eta = eta - np.inner(eta, xi)*xi
    
    eta = eta/np.linalg.norm(eta)

    return xi, eta
    
    
    
    
    
