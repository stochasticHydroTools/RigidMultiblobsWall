'''
Integrator for several rigid bodies.
'''
import numpy as np
import math as m
import scipy.sparse.linalg as spla
from functools import partial

from quaternion import Quaternion

class QuaternionIntegrator(object):
  '''
  Integrator that timesteps using deterministic forwars Euler scheme.
  '''
  
  def __init__(self, bodies, scheme): 
    ''' 
    Init object 
    '''
    self.bodies = bodies
    self.scheme = scheme
    return 

  def advance_time_step(self, dt):
    '''
    Advance time step with integrator self.scheme
    '''
    return getattr(self, self.scheme)(dt)
    

  def deterministic_forward_euler(self, dt): 
    ''' 
    Take a time step of length dt using the 
    deterministic forward Euler scheme. 
    ''' 
    print 'Integrator starting' 
    while True: 
      # print 'bodies\n', self.bodies[0].reference_configuration
      # Compute velocities 

      # Update positions 

      # Check positions if valid return 
      return
      
      
    
    




