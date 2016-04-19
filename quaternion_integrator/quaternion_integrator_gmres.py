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
               #linear_operator = None,
               #get_vectors = None,
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
    ## To save velocities and rotations
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
    
    
    #TODO, if we use location, check that we have a force calculator and
    # an iniital location.
    
    self.rf_delta = 1e-8  # delta for RFD term in RFD step

    #TODO: Make this dynamic
    self.kT = 1.0

    # Set up a check to satisfy at every step.  This needs to be 
    # overwritten manually to use it.
    #  Check function should take a state (location, orientation) if
    # has_location=True or 
    # (orietnation) if has_location = False
    #  If this is false at the end of a step, the integrator will 
    # re-take that step. This is a function that returns true or false. 
    # Can also be None, which will not check anything.  The integrator 
    # will count the total number of rejected steps.
    self.check_function = None
   
    self.rejections = 0
    self.successes = 0

    # Accumulate total velocity and angular velocity.
    self.avg_velocity = 0.0
    self.avg_omega = 0.0

  def rfd_time_step(self, dt):
    ''' Take a timestep of length dt using the RFD method '''
    # Attempt steps until we find a valid endpoint.  
    #print "self.dim =" , self.dim
    while True:
      self.veltot = []
      self.omegatot = []
      self.mob_coeff = []
      if self.has_location:
	
	
	print "self.location = "
        print self.location
        
	if self.solver == 1:
	  print "=================================================== "
	  print "==== Compute matrices and vectors for GMRES iteration  ====== "
	  print "=================================================== "
	  (J_tot,self_mobility_body, chol_mobility_blobs_each_body,\
             r_vectors_ite, rotation_matrix_ite) = \
             self.matrices_for_GMRES_ite(self.location, self.orientation, self.initial_config)
	elif self.solver == 2:
	  print "=================================================== "
	  print "==== Compute configs for Direct Solver iteration  ====== "
	  print "=================================================== "
	  ### IMPLEMENT ROUTINE FOR DIRECT SOLVER WHICH RETURN ROTATION MATRIX AND r_vectros
	  (r_vectors_ite, rotation_matrix_ite) = \
              self.matrices_for_direct_ite(self.location, self.orientation, self.initial_config)

        print "=================================================== "
        print "==== Compute forces and torques on rods  ====== "
        print "=================================================== "
        # force = self.force_calculator(r_vectors_ite)
        # torque = self.torque_calculator(r_vectors_ite, rotation_matrix_ite)
        random_forces_blobs = 1.0*(2.0*np.random.random(self.Nblobs*3)-1.0)*6.0*m.pi*self.A
        random_forcing_rods = self.force_slip(r_vectors_ite, rotation_matrix_ite,random_forces_blobs)
        
        force = random_forcing_rods[0:3*self.Nrods]*1.0
        torque = random_forcing_rods[3*self.Nrods:6*self.Nrods]*0.0
        
 
    
        print "=================================================== "
        print "==== Prescribe  slip on blobs  ====== "
        print "=================================================== " 
        # 0. Get slip on each blob
        #slip = self.slip_velocity(r_vectors_ite,self.location)
        
        
        if self.solver == 2:
	  print "=================================================== "
	  print "==== Direct Solver  ====== "
	  print "=================================================== " 
          (mobility, res_blobs)  = self.mobility(r_vectors_ite, rotation_matrix_ite)
	  #resistance = self.resistance_blobs(r_vectors_ite)
	  #lambda_tilde = np.dot(res_blobs, slip)
	  #force_slip = -self.force_slip(r_vectors_ite, rotation_matrix_ite ,lambda_tilde)
	  force = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	  torque = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
	  
	  force_deterministic = np.concatenate([force, torque]) #+ force_slip
	  
	  
	  
	  # Add noise
	  noise = np.random.normal(0.0, 1.0, self.dim*6)
	  # If mobility is positive definite use Cholesky
	  # noise_term = np.sqrt(2.0*self.kT/dt) * np.dot(mobility_half, noise) 
	  
	  # If mobility is not positive definite use eigenvectors and eigenvalues
	  eig_w, eig_v = np.linalg.eigh(mobility)
	  eig_w_sqrt_noise = np.zeros( self.dim*6 )
	  for i in range(self.dim*6):
	    if(eig_w[i] < 0):
	      eig_w_sqrt_noise[i] = 0
	    else:
	      eig_w_sqrt_noise[i] = np.sqrt(eig_w[i]) * noise[i]
	  noise_term = np.sqrt(2.0*self.kT/dt) * np.dot( eig_v, eig_w_sqrt_noise)

	  # Compute deterministic velocity
	  velocity_deterministic = np.dot(mobility, force_deterministic)
	  
	  # Calculate RFD location.
	  #rfd_noise = np.random.normal(0.0, 1.0, self.dim*6)
	  # NEED TO MAKE IT WORK TO AVOID A LOOP ON POSITION!!!
	  #rfd_location = self.location + self.rf_delta * rfd_noise[0:3*self.dim]

	  # Update each quaternion at a time for RFD orientation.
	  #rfd_location = []
	  #rfd_orientation = []
	  #for i in range(self.dim):
	    #rfd_location.append(self.location[i] + self.rf_delta*rfd_noise[3*i:3*(i+1)]) 
	    #quaternion_dt = Quaternion.from_rotation(( self.rf_delta * rfd_noise[(3*self.dim + i*3):(3*self.dim + i*3+3)] ))
	    #rfd_orientation.append(quaternion_dt*self.orientation[i])
	  
	  # divergence term d_x(N) : \Psi^T 
	  #divergence_term = self.kT*np.dot((self.mobility(rfd_location, rfd_orientation) - mobility), rfd_noise/self.rf_delta)

	  # Compute total velocity
	  ### HERE I HAVE COMMENTED THE DIV TERM 
	  velocity_and_omega = velocity_deterministic # + noise_term # + divergence_term
        
        elif self.solver == 1:

	  print "=================================================== "
	  print "==== GMRES WITH PRECONDITIONER ====== "
	  print "=================================================== "
	  linear_operator_partial = partial(self.linear_operator,\
					    r_vectors = r_vectors_ite,\
					    rotation_matrix = rotation_matrix_ite,\
					    Nbody=self.Nrods,\
					    Ncomp_blobs=self.Nblobs*3)
	  Size_system = self.Nblobs*3 + self.Nrods*6	
	  #print "Size_system = ", Size_system
	  A = spla.LinearOperator((Size_system,Size_system),\
				  matvec = linear_operator_partial,\
				  dtype='float64')


	  precond_partial = partial(self.precond,\
				    K_matrix=J_tot,\
				    mob_chol_blobs=chol_mobility_blobs_each_body,\
				    self_mob_body= self_mobility_body)
					  
	  P_optim = spla.LinearOperator( (Size_system,Size_system),\
					matvec = precond_partial,\
					dtype='float64' )				
					  
	  #RHS = np.concatenate([slip, -np.concatenate([force, torque])])
	  RHS = np.concatenate([np.zeros(self.Nblobs*3), -np.concatenate([force, torque])])

	  (sol_precond,info_precond) = spla.gmres(A,RHS,x0=self.first_guess,tol=1e-8,M=P_optim,callback=make_callback())
	  self.first_guess = sol_precond
	  
	  ## I CAN'T USE KRYPY ON PYTHON 2.6 BUT BOOST ONLY WORKS ON PYTHON 2.6
	  #print "Krypy, GMRES, Right PC "   
	  #LS = kpls.LinearSystem(A,RHS,\
				#M=None, Minv=None, Ml=None, Mr=P)
	  #solver = kpls.Gmres(LS, x0 = None, tol = 1e-8)

	  velocity_and_omega = sol_precond[self.Nblobs*3:self.Nblobs*3 + self.Nrods*6]
	  
        
        # Unpack linear and angular velocities
        velocity = velocity_and_omega[0:(3*self.dim)]
        omega = velocity_and_omega[(3*self.dim):(6*self.dim)]
        print "velocity = "
        print velocity
	print "omega = "
        print omega
        
        self.avg_velocity += np.linalg.norm(velocity)
        self.avg_omega += np.linalg.norm(omega)
        

        new_location = []
        # Update location and save velocity and rotation
        for i in range(self.dim): 
	  # TO UNCOMMENT
          #new_location.append(self.location[i] + dt*velocity[3*i:3*(i+1)])
          
          self.veltot.append(velocity[3*i:3*(i+1)])
          self.omegatot.append(omega[3*i:3*(i+1)])
          
          ## TO COMMENT, ONLY FOR HYDRO TESTS
          # self.mob_coeff.append(mobility)

          ## TO COMMENT, ONLY FOR HYDRO TESTS
          new_location.append(self.location[i] + 0.01*np.array([0.0, 0.0, (-1.0)**(float(i+1))]))
	

      # BD: I DID NOT MODIFY THIS PART SINCE WE INTEGRATE POSITIONS
      else:
        rfd_noise = np.random.normal(0.0, 1.0, self.dim*3)
        mobility  = self.mobility(self.orientation)
        mobility_half = np.linalg.cholesky(mobility)
        torque = self.torque_calculator(self.orientation)
        
        noise = np.random.normal(0.0, 1.0, self.dim*3)
        # Update each quaternion at a time for rfd orientation.
        rfd_orientation = []
        for i in range(self.dim):
          quaternion_dt = Quaternion.from_rotation((self.rf_delta * rfd_noise[(i*3):(i*3+3)]))
          rfd_orientation.append(quaternion_dt*self.orientation[i])

        # divergence term d_x(M) : \Psi^T 
        divergence_term = self.kT*np.dot((self.mobility(rfd_orientation) - mobility), rfd_noise/self.rf_delta)
        omega = (np.dot(mobility, torque) + np.sqrt(2.*self.kT/dt) * np.dot(mobility_half, noise) + divergence_term)

 
      # For with location and without location, we update orientation the same way.
      new_orientation = []
      for i in range(self.dim):
	# TO UNCOMMENT
        #quaternion_dt = Quaternion.from_rotation((omega[(i*3):(i*3+3)])*dt)
        #new_orientation.append(quaternion_dt*self.orientation[i])
        
        # TO COMMENT, THIS IS ONLY FOR HYDRO TESTS
        new_orientation.append(self.orientation[i])

      # Check validity of new state.
      if self.has_location:
	# TO UNCOMMENT
        #if self.check_new_state(new_location, new_orientation):
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
    
