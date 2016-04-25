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
  def __init__(self, bodies, Nblobs, scheme): 
    ''' 
    Init object 
    '''
    self.bodies = bodies
    self.Nblobs = Nblobs
    self.scheme = scheme

    # Other variables
    self.get_blobs_r_vectors = None
    self.mobility_blobs = None
    self.force_torque_calculator = None
    self.calc_K_matrix = None
    self.eta = None
    self.a = None
    

    # Optional variables
    self.calc_slip = None
    self.calc_force_torque = None
    self.mobility_blobs = None
    self.mobility_bodies = None
    
    return 

  def advance_time_step(self, dt):
    '''
    Advance time step with integrator self.scheme
    '''
    return getattr(self, self.scheme)(dt)
    

  def deterministic_forward_euler(self, dt): 
    ''' 
    Take a time step of length dt using the deterministic forward Euler scheme. 
    The function uses gmres to solve the rigid body equations.
    ''' 
    print 'Integrator starting (gmres)' 
    while True: 
      # print 'bodies\n', self.bodies[0].reference_configuration
      # Compute velocities 

      # Update positions 

      # Check positions if valid return 
      return
      

  def deterministic_forward_euler_dense_algebra(self, dt): 
    ''' 
    Take a time step of length dt using the deterministic forward Euler scheme. 
    The function uses dense algebra methods to solve the equations.
    ''' 
    print 'Integrator starting (dense algebra)' 
    while True: 
      # Calculate slip on blobs
      if self.calc_slip is not None:
        slip = self.calc_slip(self.bodies, self.Nblobs)
      else:
        slip = np.zeros((self.Nblobs, 3))

      # Get blobs coordinates
      r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)

      # Calculate mobility (M) at the blob level
      mobility_blobs = self.mobility_blobs(r_vectors_blobs, self.eta, self.a)

      # Calculate resistance at the blob level (use np.linalg.inv or np.linalg.pinv)
      resistance_blobs = np.linalg.inv(mobility_blobs)

      # Calculate constraint force due to slip l = M^{-1}*slip
      force_slip = np.dot(resistance_blobs, np.reshape(slip, (3*self.Nblobs,1)))

      # Calculate force-torque on bodies
      force_torque = self.force_torque_calculator(self.bodies, r_vectors_blobs)

      # Calculate block-diagonal matrix K
      K = self.calc_K_matrix(self.bodies, self.Nblobs)
     
      # Calculate mobility (N) at the body level. Use np.linalg.inv or np.linalg.pinv
      resistance_bodies = np.dot(K.T, np.dot(resistance_blobs, K))
      mobility_bodies = np.linalg.inv(np.dot(K.T, np.dot(resistance_blobs, K)))

      # Compute velocities
      velocities = np.dot(mobility_bodies, np.reshape(force_torque, 6*len(self.bodies)))

      # Update location orientation

      # Check positions, if valid return

      

      return
      
      
  
    




