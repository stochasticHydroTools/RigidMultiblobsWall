'''
Simple quaternion integrator.  
'''
import numpy as np

class Quaternion(object):
  
  def __init__(self, entries):
    ''' Constructor, takes 4 entries = s, p1, p2, p3 as a numpy array. '''
    self.entries = entries

  @classmethod
  def FromRotation(cls, phi):
    ''' Create a quaternion given an angle of rotation phi,
    which represents a rotation clockwise about the vector phi of magnitude 
    phi. This will be used with phi = omega*dt or similar in the integrator.'''
    phi_norm = np.linalg.norm(phi)
    s = np.array([np.cos(phi_norm/2.)])
    p = np.sin(phi_norm/2)*(phi/phi_norm)
    return cls(np.concatenate([s, p]))

    
  def __mul__(self, other):
    ''' 
    Quaternion multiplication.  In this case, other is the 
    right quaternion. 
    '''
    s = (self.GetS()*other.GetS() - 
         np.dot(self.GetP(), other.GetP()))
    p = (self.GetS()*other.GetP() + other.GetS()*self.GetP()
         - np.cross(self.GetP(), other.GetP()))
    
    return Quaternion(np.concatenate(([s], p)))
  
  
  def GetP(self):
    ''' Get the p vector, last 3 entries. '''
    #TODO: Figure out how to use params decorator.
    return self.entries[1:4]

  
  def GetS(self):
    ''' Return scalar s, the first entry. '''
    #TODO: Figure out how to use params decorator.
    return self.entries[0]


