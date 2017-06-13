'''
Simple integrator for N quaternions.
'''
import numpy as np
import math as m
from quaternion import Quaternion
import scipy.sparse.linalg as spla
## I CAN'T USE KRYPY ON PYTHON 2.6 BUT BOOST ONLY WORKS ON PYTHON 2.6
#import krypy.linsys as kpls

from functools import partial

class QuaternionIntegratorGMRES(object):
  '''
  Integrator that timesteps using Fixman quaternion updates.
  '''
  def __init__(self, 
               mobility, 
               initial_orientation, 
               torque_calculator, 
               has_location = False, 
               initial_location = None, 
               force_calculator = None,
               slip_velocity = None,
               resistance_blobs = None,
               mobility_blobs = None,
               mobility_vector_prod = None,
               blob_vel = None,
               force_slip = None):
    '''
    Set up components of the integrator.  
    args: 
    
      mobility: function that takes a vector of positions (quaternions) 
                and returns a matrix of the mobility evaluated there.

      torque_calculator: function that takes a vector of positions
                         (quaternions) and returns the torque evaluated there as 
                         a numpy array where the first three components are the 
                         torque on the first quaternion, etc.

      initial_orientation: vector of quaternions representing the initial 
                           configuration of the system.
    
      has_location: boolean indicating whether we keep location
                    as well as orientation.
    '''
    self.mobility = mobility
    # BD: DIM IS THE NUMBER OF QUATERNIONS, FOR ONE BODY
    self.dim = len(initial_orientation)
    self.torque_calculator = torque_calculator
    self.orientation = initial_orientation
    self.has_location = has_location
    self.location = initial_location
    self.force_calculator = force_calculator
    self.slip_velocity = slip_velocity
    self.resistance_blobs = resistance_blobs
    self.mobility_blobs = mobility_blobs
    self.mobility_vector_prod = mobility_vector_prod
    self.blob_vel = blob_vel
    self.force_slip = force_slip
    # To save velocities and rotations
    self.veltot = []
    self.omegatot = []
    self.mob_coeff = []
    
    self.Nblobs = None
    self.Nrods = None
    self.linear_operator = None
    self.precond = None
    self.get_vectors = None
    self.matrices_for_GMRES_ite  = None
    self.matrices_for_direct_ite  = None
    self.first_guess  = None
    self.initial_config  = None
    self.A = None     
    self.rf_delta = 1e-8  # delta for RFD term in RFD step
    self.kT = 1.0

    # Set up a check to satisfy at every step.  This needs to be 
    # overwritten manually to use it.
    # Check function should take a state (location, orientation) if
    # has_location=True or (orietnation) if has_location = False
    # If this is false at the end of a step, the integrator will 
    # re-take that step. This is a function that returns true or false. 
    # Can also be None, which will not check anything. The integrator 
    # will count the total number of rejected steps.
    self.check_function = None   
    self.rejections = 0
    self.successes = 0


  def deterministic_forward_euler_time_step(self, dt):
    ''' Take a timestep of length dt using the RFD method '''
    # Attempt steps until we find a valid endpoint.  
    while True:
      self.veltot = []
      self.omegatot = []
      self.mob_coeff = []
      if self.has_location:
        (J_tot,self_mobility_body, chol_mobility_blobs_each_body,\
           r_vectors_ite, rotation_matrix_ite) = \
           self.matrices_for_GMRES_ite(self.location, self.orientation, self.initial_config)

        force = self.force_calculator(r_vectors_ite)
        torque = self.torque_calculator(r_vectors_ite, rotation_matrix_ite)

        # Get slip on each blob
        slip = self.slip_velocity(r_vectors_ite, self.location)            

        # Set linear operators
        linear_operator_partial = partial(self.linear_operator,\
					    r_vectors = r_vectors_ite,\
					    rotation_matrix = rotation_matrix_ite,\
					    Nbody=self.Nrods,\
					    Ncomp_blobs=self.Nblobs*3)

	Size_system = self.Nblobs*3 + self.Nrods*6	
	A = spla.LinearOperator((Size_system,Size_system), matvec = linear_operator_partial, dtype='float64')
	precond_partial = partial(self.precond,\
				    K_matrix=J_tot,\
				    mob_chol_blobs=chol_mobility_blobs_each_body,\
				    self_mob_body= self_mobility_body)
					  
	P_optim = spla.LinearOperator((Size_system,Size_system),\
                                        matvec = precond_partial,\
                                        dtype='float64' )				
					  
        # Set right hand side
	RHS = np.concatenate([slip, -np.concatenate([force, torque])])

        # Solve preconditioned linear system
	(sol_precond, info_precond) = spla.gmres(A, RHS,x0=self.first_guess, tol=1e-8, M=P_optim, callback=make_callback())
	self.first_guess = sol_precond  

        # Get bodies velocities
        velocity_and_omega = sol_precond[self.Nblobs*3:self.Nblobs*3 + self.Nrods*6]  
        
        # Unpack linear and angular velocities
        velocity = velocity_and_omega[0:(3*self.dim)]
        omega = velocity_and_omega[(3*self.dim):(6*self.dim)]
        
        new_location = []
        # Update location and save velocity and rotation
        for i in range(self.dim): 
          new_location.append(self.location[i] + dt*velocity[3*i:3*(i+1)])         
          self.veltot.append(velocity[3*i:3*(i+1)])
          self.omegatot.append(omega[3*i:3*(i+1)])
          
      else:
        raise Exception('ERROR, algorithm for bodies without location is not implemented')
 
      # For with location and without location, we update orientation the same way.
      new_orientation = []
      for i in range(self.dim):
        quaternion_dt = Quaternion.from_rotation((omega[(i*3):(i*3+3)])*dt)
        new_orientation.append(quaternion_dt*self.orientation[i])
        
      # Check validity of new state.
      if self.has_location:
        if self.check_new_state(new_location, new_orientation):
          self.location = new_location
          self.orientation = new_orientation
          self.successes += 1
          return
      else:
        if self.check_new_state(None, new_orientation):
          self.orientation = new_orientation
          self.successes += 1
          return  
          
  def check_new_state(self, location, orientation):
    ''' 
    Use the check function to test if the new state is valid.
    If not, the timestep will be thrown out. 
    '''
    if self.check_function:
      if self.has_location:
        admissible = self.check_function(location, orientation,self.initial_config)
      else:
        admissible = self.check_function(orientation,self.initial_config)
      if not admissible:
        self.rejections += 1
        print 'rejections =', self.rejections
      return admissible
    else:
      return True

# Callback generator
def make_callback():
    closure_variables = dict(counter=0, residuals=[]) 
    def callback(residuals):
        closure_variables["counter"] += 1
        closure_variables["residuals"].append(residuals)
        print closure_variables["counter"], residuals
    return callback
    
