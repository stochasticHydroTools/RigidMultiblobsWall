'''
Simple integrator for N quaternions.
'''
import numpy as np
import math as m
import time
from quaternion import Quaternion
from stochastic_forcing import stochastic_forcing as sf
import scipy.sparse.linalg as spla
## I CAN'T USE KRYPY ON PYTHON 2.6 BUT BOOST ONLY WORKS ON PYTHON 2.6
#import krypy.linsys as kpls

from functools import partial

class PCQuaternionIntegratorGMRES(object):
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
	       K_matrix_T_vector_prod = None,
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
    self.K_matrix_T_vector_prod = K_matrix_T_vector_prod
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

    self.ISstochastic = None


  def stochastic_PC_time_step(self, dt):
    ''' Take a timestep of length dt using the RFD method '''
    # Attempt steps until we find a valid endpoint.  
    while True:
      self.veltot = []
      self.omegatot = []
      self.mob_coeff = []
      if self.has_location:
        (J_tot_P,self_mobility_body_P, chol_mobility_blobs_each_body_P,\
           r_vectors_ite_P, rotation_matrix_ite_P) = \
           self.matrices_for_GMRES_ite(self.location, self.orientation, self.initial_config)

        force_P = self.force_calculator(r_vectors_ite_P)
        torque_P = self.torque_calculator(r_vectors_ite_P, rotation_matrix_ite_P)

	# Generate W increments for predictor step and part of corrector step
	if self.ISstochastic:
	  	#startTime = time.time()
	  W1 = np.random.normal(0.0, 1.0, self.Nblobs*3)
	  W2 = np.random.normal(0.0, 1.0, self.Nblobs*3)
	  MnW2 = self.mobility_vector_prod(W2, r_vectors_ite_P)
	  Mn = self.mobility_blobs(r_vectors_ite_P) 
          	#print time.time() - startTime



        # Get slip on each blob
        slip_P = -1.0*self.slip_velocity(r_vectors_ite_P, self.location)  

	if self.ISstochastic:
	  	#startTime = time.time()
	  #rand_slip_P = np.dot(chol_mobility_blobs_each_body_P,W1)
	  rand_slip_P = sf.stochastic_forcing_eig(Mn,1.0,W1)

	  #mobility_vector_prod_partial_P = partial(self.mobility_vector_prod, r_vectors = r_vectors_ite_P)
	  #(rand_slip_P,max_iter_P) = sf.stochastic_forcing_lanczos(factor = 1,\
          #tolerance = 1e-10,\
          #max_iter = 1000,\
          #name = '',\
          #dim = None,\
          #mobility = None,\
          #mobility_mult = mobility_vector_prod_partial_P,\
          #z = W1)
	  slip_P = slip_P - np.sqrt(4.0*self.kT/dt)*(rand_slip_P + W2) # need this negative because of the -K^{T} in the lin. op.
	  	#print time.time() - startTime
	   

        # Set linear operators
        linear_operator_partial_P = partial(self.linear_operator,\
					    r_vectors = r_vectors_ite_P,\
					    rotation_matrix = rotation_matrix_ite_P,\
					    Nbody=self.Nrods,\
					    Ncomp_blobs=self.Nblobs*3)

	Size_system = self.Nblobs*3 + self.Nrods*6	
	A_P = spla.LinearOperator((Size_system,Size_system), matvec = linear_operator_partial_P, dtype='float64')
	precond_partial_P = partial(self.precond,\
				    K_matrix=J_tot_P,\
				    mob_chol_blobs=chol_mobility_blobs_each_body_P,\
				    self_mob_body= self_mobility_body_P)
					  
	P_optim_P = spla.LinearOperator((Size_system,Size_system),\
                                        matvec = precond_partial_P,\
                                        dtype='float64' )				
					  
        # Set right hand side
	RHS_P = np.concatenate([slip_P, -np.concatenate([force_P, torque_P])])

        # Solve preconditioned linear system
	(sol_precond_P, info_precond_P) = spla.gmres(A_P, RHS_P,x0=self.first_guess, tol=1e-8, M=P_optim_P, callback=make_callback())
	self.first_guess = sol_precond_P  

        # Get bodies velocities
        velocity_and_omega_P = sol_precond_P[self.Nblobs*3:self.Nblobs*3 + self.Nrods*6]  
        
        # Unpack linear and angular velocities
        velocity_P = velocity_and_omega_P[0:(3*self.dim)]
        omega_P = velocity_and_omega_P[(3*self.dim):(6*self.dim)]
        
        mid_location = []
        # Update location and save velocity and rotation
        for i in range(self.dim): 
          mid_location.append(self.location[i] + 0.5*dt*velocity_P[3*i:3*(i+1)])         
          #self.veltot.append(velocity[3*i:3*(i+1)])
          #self.omegatot.append(omega[3*i:3*(i+1)])

	mid_orientation = []
        for i in range(self.dim):
          quaternion_dt = Quaternion.from_rotation((omega_P[(i*3):(i*3+3)])*(0.5*dt))
          mid_orientation.append(quaternion_dt*self.orientation[i])

	#
	#########################################
	# Corrector step
	#########################################
	#

	(J_tot,self_mobility_body, chol_mobility_blobs_each_body,\
           r_vectors_ite, rotation_matrix_ite) = \
           self.matrices_for_GMRES_ite(mid_location, mid_orientation, self.initial_config)

	#print np.linalg.norm(np.array(mid_location)-np.array(self.location))
	#print np.linalg.norm(np.array(r_vectors_ite)-np.array(r_vectors_ite_P))

        force = self.force_calculator(r_vectors_ite)
        torque = self.torque_calculator(r_vectors_ite, rotation_matrix_ite)

	#print np.linalg.norm(np.array(force)-np.array(force_P))

	# Generate W increments for corrector step
	if self.ISstochastic:
	  W3 = np.random.normal(0.0, 1.0, self.Nblobs*3)
	  Wcorector = W1 # should be W1 + W3 but just W1 seems to work

        # Get slip on each blob
        slip = -1.0*self.slip_velocity(r_vectors_ite, mid_location)  

	#print np.linalg.norm(np.array(slip))

	# compute (M^(n))^(1/2)[W1 + W3]
	if self.ISstochastic:
	  	#startTime = time.time()
	  #rand_slip_C = np.dot(chol_mobility_blobs_each_body_P,Wcorector)
	  rand_slip_C = sf.stochastic_forcing_eig(Mn,1.0,Wcorector)

	  #(rand_slip_C,max_iter) = sf.stochastic_forcing_lanczos(factor = 1,\
          #tolerance = 1e-10,\
          #max_iter = 1000,\
          #name = '',\
          #dim = None,\
          #mobility = None,\
          #mobility_mult = mobility_vector_prod_partial_P,\
          #z = Wcorector)
	  slip = slip - np.sqrt(1.0*self.kT/dt)*(rand_slip_C - MnW2)   # need this negative because of the -K^{T} in the lin. op.
	  	#print time.time() - startTime

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
	# compute (K^(n))^T [W2]
	FT = np.concatenate([force, torque])
	if self.ISstochastic:
	  RandFT = self.K_matrix_T_vector_prod(r_vectors_ite_P, rotation_matrix_ite_P, W2)
	  FT = FT + np.sqrt(1.0*self.kT/dt)*RandFT

	RHS = np.concatenate([slip, -FT])

        # Solve preconditioned linear system
	(sol_precond, info_precond) = spla.gmres(A, RHS,x0=self.first_guess, tol=1e-8, M=P_optim, callback=make_callback())

	#print np.linalg.norm(np.array(sol_precond)-np.array(sol_precond_P))
	self.first_guess = sol_precond  

        # Get bodies velocities
        velocity_and_omega = sol_precond[self.Nblobs*3:self.Nblobs*3 + self.Nrods*6]  
        
        # Unpack linear and angular velocities
        velocity = velocity_and_omega[0:(3*self.dim)]
        omega = velocity_and_omega[(3*self.dim):(6*self.dim)]

	#print np.linalg.norm(np.array(velocity)-np.array(velocity_P))
        
        new_location = []
        # Update location and save velocity and rotation
        for i in range(self.dim): 
          new_location.append(self.location[i] + dt*velocity[3*i:3*(i+1)])         
          self.veltot.append(velocity[3*i:3*(i+1)])
          self.omegatot.append(omega[3*i:3*(i+1)])
	
	new_orientation = []
        for i in range(self.dim):
          quaternion_dt = Quaternion.from_rotation((omega[(i*3):(i*3+3)])*dt)
          new_orientation.append(quaternion_dt*self.orientation[i])
          
	#print np.linalg.norm(np.array(new_location)-np.array(self.location)) - np.linalg.norm(dt*np.array(velocity))
	#print np.linalg.norm(np.array(new_location)-np.array(mid_location))
      else:
        raise Exception('ERROR, algorithm for bodies without location is not implemented')
        
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
        #print closure_variables["counter"], residuals
    return callback
    
