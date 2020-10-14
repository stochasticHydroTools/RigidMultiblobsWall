'''
Simple quaternion object for use with quaternion integrators.
'''

import numpy as np

class Quaternion(object):
  
  def __init__(self, entries):
    ''' Constructor, takes 4 entries = s, p1, p2, p3 as a numpy array. '''
    self.entries = entries
    self.s = np.array(entries[0])
    self.p = np.array(entries[1:4])


  @classmethod
  def from_rotation(cls, phi):
    ''' Create a quaternion given an angle of rotation phi,
    which represents a rotation clockwise about the vector phi of magnitude 
    phi. This will be used with phi = omega*dt or similar in the integrator.'''
    phi_norm = np.linalg.norm(phi)
    s = np.array([np.cos(phi_norm/2.)])
    if phi_norm != 0:
      p = np.sin(phi_norm/2)*(phi/phi_norm)
    else:
      p = np.zeros(3)
    return cls(np.concatenate([s, p]))

    
  def __mul__(self, other):
    ''' 
    Quaternion multiplication.  In this case, other is the 
    right quaternion. 
    '''
    s = (self.s*other.s - 
         np.dot(self.p, other.p))
    p = (self.s*other.p + other.s*self.p
         + np.cross(self.p, other.p))
    return Quaternion(np.concatenate(([s], p)))


  def rotation_matrix(self):
    ''' 
    Return the rotation matrix representing rotation
    by this quaternion.
    '''
    # Cross product matrix for p, actually the negative.
    diag = self.s**2 - 0.5
    return 2.0 * np.array([[self.p[0]**2+diag,                    self.p[0]*self.p[1]-self.s*self.p[2], self.p[0]*self.p[2]+self.s*self.p[1]], 
                           [self.p[1]*self.p[0]+self.s*self.p[2], self.p[1]**2+diag,                    self.p[1]*self.p[2]-self.s*self.p[0]],
                           [self.p[2]*self.p[0]-self.s*self.p[1], self.p[2]*self.p[1]+self.s*self.p[0], self.p[2]**2+diag]])


  def __str__(self):
    return '[ %f, %f, %f, %f ]' % (self.s, self.p[0], self.p[1], self.p[2])


  def inverse(self):
    ''' Return the inverse quaternion.'''
    return Quaternion([self.s, -1.*self.p[0], -1.*self.p[1],
                       -1.*self.p[2]])

  def square_root(self):
    ''' Return the root quaternion.'''
    if self.s != -1:
      return Quaternion([np.sqrt((self.s+1.0)/2.0), np.sqrt(1.0/(2.0*self.s+2.0))*self.p[0], np.sqrt(1.0/(2.0*self.s+2.0))*self.p[1],
                      np.sqrt(1.0/(2.0*self.s+2.0))*self.p[2]])
    else:
      return Quaternion([0.0, 0.0, 0.0, 1.0])


  def rotation_angle(self):
    ''' Return 3 dimensional rotation angle that the quaternion represents. '''
    phi_norm = 2.*np.arccos(self.s)
    return phi_norm*self.p/(np.linalg.norm(self.p))

  def random_orientation(self):
    '''Give this quaternion object a random orientation'''
    theta = np.random.normal(0., 1., 4)
    theta = theta/np.linalg.norm(theta)
    self.entries = theta
    self.s = theta[0]
    self.p = theta[1:4]
    
