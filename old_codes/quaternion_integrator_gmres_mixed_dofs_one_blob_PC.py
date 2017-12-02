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
    self.blob_vel = blob_vel
    self.force_slip = force_slip
    ## To save velocities and rotations
    self.veltot = []
    self.omegatot = []
    self.mob_coeff = []
    
    self.Nblobs = None
    self.Nrods = None
    self.linear_operator = None
    #self.mobility_tt_prod = None
    self.mobility_rr_prod = None
    self.mobility_rt_prod = None
    self.mobility_tt_tr_prod = None
    
    self.precond = None
    self.get_vectors = None
    self.matrices_for_GMRES_ite  = None
    self.matrices_for_direct_ite  = None
    self.first_guess  = None
    self.velocity_previous  = None
    self.initial_config  = None
    self.A = None
    self.eta = None
    self.constant_torque = None
    self.no_Fz = None
    self.no_Fy = None
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

  def fixman_time_step(self, dt):
    ''' Take a timestep of length dt using the Fixman method '''
    # Try to take steps until one is valid
    while True:
      if self.has_location:
        # Handle integrator with location as well.
        mobility  = self.mobility(self.location, self.orientation)
        mobility_half = np.linalg.cholesky(mobility)
        noise = np.random.normal(0.0, 1.0, self.dim*6)
        force = self.force_calculator(self.location, self.orientation)
        torque = self.torque_calculator(self.location, self.orientation)
        velocity_and_omega = (np.dot(mobility, np.concatenate([force, torque])) +
                              np.sqrt(4.0*self.kT/dt)*
                              np.dot(mobility_half, noise))
        velocity = velocity_and_omega[0:(3*self.dim)]
        omega = velocity_and_omega[(3*self.dim):(6*self.dim)]
      else:
        mobility  = self.mobility(self.orientation)
        mobility_half = np.linalg.cholesky(mobility)
        noise = np.random.normal(0.0, 1.0, self.dim*3)
        torque = self.torque_calculator(self.orientation)
        omega = (np.dot(mobility, torque) + 
               np.sqrt(4.0*self.kT/dt)*np.dot(mobility_half, noise))
        
      # Update each quaternion at a time.
      orientation_midpoint = []
      for i in range(self.dim):
        quaternion_dt = Quaternion.from_rotation((omega[(i*3):(i*3+3)])*dt/2.)
        orientation_midpoint.append(quaternion_dt*self.orientation[i])

      if self.has_location:
        location_midpoint = self.location + 0.5*dt*velocity
        if not self.check_new_state(location_midpoint, orientation_midpoint):
          # restart the step.
          continue

      if self.has_location:
        mobility_tilde = self.mobility(location_midpoint, orientation_midpoint)
        noise = noise + np.random.normal(0.0, 1.0, self.dim*6)
        force_tilde = self.force_calculator(location_midpoint, orientation_midpoint)
        torque_tilde = self.torque_calculator(location_midpoint, orientation_midpoint)
        mobility_half_inv = np.linalg.inv(mobility_half)
        velocity_and_omega_tilde = (
          np.dot(mobility_tilde, 
                 np.concatenate([force_tilde, torque_tilde])) + np.sqrt(self.kT/dt)*
          np.dot(mobility_tilde, np.dot(mobility_half_inv.T, noise)))
        velocity_tilde = velocity_and_omega_tilde[0:(3*self.dim)]
        self.avg_velocity += np.linalg.norm(velocity_tilde)
        omega_tilde = velocity_and_omega_tilde[(3*self.dim):(6*self.dim)]
        self.avg_omega += np.linalg.norm(omega_tilde)
      
      else:
        mobility_tilde = self.mobility(orientation_midpoint)
        noise = noise + np.random.normal(0.0, 1.0, self.dim*3)
        torque_tilde = self.torque_calculator(orientation_midpoint)
        mobility_half_inv = np.linalg.inv(mobility_half)
        omega_tilde = (
          np.dot(mobility_tilde, torque_tilde) + np.sqrt(self.kT/dt)*
          np.dot(mobility_tilde, np.dot(mobility_half_inv.T, noise)))
        
      new_orientation = []
      for i in range(self.dim):
        quaternion_dt = Quaternion.from_rotation((omega_tilde[(i*3):(i*3+3)])*dt)
        new_orientation.append(quaternion_dt*self.orientation[i])

      # Check that the new state is admissible. Re-take the step 
      # (with tail recursion) if the new state is invalid.
      if self.has_location:
        new_location = self.location + dt*velocity_tilde
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


  def rfd_time_step(self, dt, current_ite):
    ''' Take a timestep of length dt using the RFD method '''
    # Attempt steps until we find a valid endpoint.  
    #print "self.dim =" , self.dim
    self.veltot = []
    self.omegatot = []
    self.mob_coeff = []
    
    
    
    if self.has_location:

      #print self.location
      #print "=================================================== "
      #print "==== Compute forces and torques on rods  ====== "
      #print "=================================================== "
      force = self.force_calculator(self.location)
      #force = np.zeros(self.Nrods*3)
      if self.no_Fz == 1 and self.no_Fy == 1:
	for n in range(self.Nrods):
	  force[3*n+1] = 0.0
	  force[3*n+2] = 0.0
      elif self.no_Fz == 1 and self.no_Fy == 0:
	for n in range(self.Nrods):
	  force[3*n+2] = 0.0

      #torque = self.torque_calculator(r_vectors_ite, rotation_matrix_ite)
    
      #random_forces_blobs = 1.0*(2.0*np.random.random(self.Nblobs*3)-1.0)*6.0*m.pi*self.A
      #random_forcing_rods = self.force_slip(r_vectors_ite, random_forces_blobs)
      
      #force = random_forcing_rods[0:3*self.Nrods]*0.0
      #torque = random_forcing_rods[3*self.Nrods:6*self.Nrods]*0.0
      #force =  1.0*(2.0*np.random.random(self.Nrods*3)-1.0)*6.0*m.pi*0.225*self.eta*0.0
      #print "force = ", force
      #raw_input()
      
      
      #print "=================================================== "
      #print "==== Noise term  ====== "
      #print "=================================================== "
      if self.kT>0:
	noise =  np.random.normal(0.0, 1.0, self.Nrods*6)
	noise_vel_omega = np.zeros( self.Nrods*6 )
	for n in range(self.Nrods):
	  noise_vel_omega[n*6:(n+1)*6] = \
	    np.dot(np.linalg.cholesky(self.mobility_blobs(self.location[n])) \
	          ,noise[n*6:(n+1)*6])
	  print "n, noise_vel_omega[n*6:(n+1)*6] = ", n, noise_vel_omega[n*6:(n+1)*6]
	  raw_input()
	noise_vel_omega = np.sqrt(2.0*self.kT/dt)*noise_vel_omega	

      #print "=================================================== "
      #print "==== Mixed DOFS  ====== "
      #print "=================================================== "
      Omega_known =np.zeros(3*self.Nrods)
      for n in range(self.Nrods):
	  Amplitude_omega =-20.0
	  #Omega_known[3*n:3*(n+1)] = np.array([Amplitude_omega*m.cos(2.0*m.pi*current_ite*dt*0.01),\
					      #Amplitude_omega*m.sin(2.0*m.pi*current_ite*dt*0.01),\
					      #0.0])
	  Omega_known[3*n:3*(n+1)] = np.array([0.0,\
					      Amplitude_omega,\
					      0.0])

      #torque =np.zeros(3*self.Nrods)
      #for n in range(self.Nrods):
	  #Amplitude_torque =-Amplitude_omega*8.0*m.pi*self.eta*self.A**3
	  ##Omega_known[3*n:3*(n+1)] = np.array([Amplitude_omega*m.cos(2.0*m.pi*current_ite*dt*0.01),\
					      ##Amplitude_omega*m.sin(2.0*m.pi*current_ite*dt*0.01),\
					      ##0.0])
	  #torque[3*n:3*(n+1)] = np.array([0.0,\
					      #Amplitude_torque,\
					      #0.0])
      #print "Omega_known = ", Omega_known
      
      #print "=================================================== "
      #print "==== Prescribe  slip on blobs  ====== "
      #print "=================================================== " 
      # 0. Get slip on each blob
      #slip = self.slip_velocity(r_vectors_ite,self.location)
    
    
      if self.constant_torque == 0:
	#print "=================================================== "
	#print "==== GMRES WITH PRECONDITIONER ====== "
	#print "=================================================== "
	#force =np.zeros(3*self.Nrods)# np.array([1e-2, 0.0, 0.0, 1e-2,0.0, 0.0, 1e-2,0.0, 0.0])*0.0
	#print "force = ", force
	RHS = Omega_known - self.mobility_rt_prod(self.location,force)
	#print RHS
	#raw_input()
	
	linear_operator_partial = partial(self.linear_operator,\
					  r_vectors = self.location)
	Size_system = self.Nrods*3	
	#print "Size_system = ", Size_system
	A = spla.LinearOperator((Size_system,Size_system),\
				matvec = linear_operator_partial,\
				dtype='float64')

					
	P_optim = spla.LinearOperator( (Size_system,Size_system),\
				      matvec = self.precond,\
				      dtype='float64' )				
					
	#RHS = np.concatenate([slip, -np.concatenate([force, torque])])
	
	#print "J_rot.shape = ", J_rot.shape
	
	#print "RHS.shape = ",RHS.shape		      
	#print "Size_system = ", Size_system	
	
	#(sol_precond,info_precond) = spla.gmres(A,RHS,x0=self.first_guess,tol=1e-6,M=P_optim,callback=make_callback())
	(sol_precond,info_precond) = spla.gmres(A,RHS,x0=self.first_guess,tol=1e-6,M=P_optim)

	self.first_guess = sol_precond
      else:
	sol_precond = 8.0*m.pi*self.eta*self.A**3*Omega_known
      #print "sol_precond = ", sol_precond
      #velocity = self.mobility_tt_tr_prod(self.location,force, torque)

      #omega =self.mobility_rr_prod(self.location,torque) + self.mobility_rt_prod(self.location,force)

      velocity = self.mobility_tt_tr_prod(self.location,force, sol_precond)
      #velocity = self.mobility_tt_tr_prod(self.location,force, torque)
      #omega = self.mobility_rr_prod(self.location, sol_precond) + self.mobility_rt_prod(self.location, force)
      omega = Omega_known
      # Unpack linear and angular velocities
      #velocity = velocity_and_omega[0:(3*self.dim)]
      #omega = velocity_and_omega[(3*self.dim):(6*self.dim)]
      #print "velocity = "
      #print velocity
      #print "omega = "
      #print omega
      #raw_input()
      #print "velocity/drag= "
      #print velocity*(6.0*m.pi*self.eta*self.A**2)
      #print "velocity/torque= "
      #print velocity*(6.0*m.pi*self.eta*self.A**2)/torque[4]
      #print "omega/torque= "
      #print omega*(8.0*m.pi*self.eta*self.A**3)/torque[4]
      #print "torque/torque_bulk = "
      #print (8.0*m.pi*self.eta*self.A**3*Omega_known[1])/torque[1]
      ##print "omega = "
      ##print omega
      #raw_input()
      
      if self.no_Fz == 1 and self.no_Fy == 1:
	for n in range(self.Nrods):
	  velocity[3*n+1] = 0.0
	  velocity[3*n+2] = 0.0
      elif self.no_Fz == 1 and self.no_Fy == 0:
	for n in range(self.Nrods):
	  velocity[3*n+2] = 0.0
	  
      self.avg_velocity += np.linalg.norm(velocity)
      self.avg_omega += np.linalg.norm(omega)
      

      self.veltot = velocity
      self.omegatot= omega

  def time_integrator_AB2AM2EM(self, dt, current_ite):
    ''' Take a timestep of length dt using the RFD method '''
    # Attempt steps until we find a valid endpoint.  
    #print "self.dim =" , self.dim
    while True:
      if self.has_location:

	#print"############### Predictor: AB2"
        self.rfd_time_step(dt,current_ite)
        new_location = []
        #print "self.location = ",self.location
	#print "self.veltot = ",self.veltot
	#print "self.veltot[0] = ",self.veltot[0]
	#print "self.velocity_previous = ",self.velocity_previous
	
	if current_ite==0:
	  
	  #print "Use Explicit Euler for 1st time Step"
	  for i in range(self.Nrods): 
	    new_location.append(self.location[i] +\
		dt*self.veltot[3*i:3*(i+1)]) 
	else:
	  
	  for i in range(self.Nrods): 
	     #print"# We now use AB2 in the absence of fluctuations"
	    new_location.append(self.location[i] +\
		dt*(1.5*self.veltot[3*i:3*(i+1)] - 0.5*self.velocity_previous[3*i:3*(i+1)]))

          # V^p(n+1)
	  self.velocity_previous = self.veltot
	  # X(n)
	  location_save = self.location
	  # X^p(n+1)
	  self.location = new_location
	    
	  #print"############### Corrector AM2 + Error monitoring"
	  self.rfd_time_step(dt,current_ite)
	  new_location = []
	  for i in range(self.Nrods): 
	      # We now use AM2 in the absence of fluctuations
	      new_location.append(location_save[i] +\
		  0.5*dt*(self.veltot[3*i:3*(i+1)] + self.velocity_previous[3*i:3*(i+1)]))
              #####  Error monitoring
	      new_location[i] = 1.0/6.0*(5.0*new_location[i]+self.location[i])
	  #print "self.location = ",self.location
	  #print "new_location = ",new_location

     
      self.velocity_previous = self.veltot
      
      # Check validity of new state.
      if self.has_location:
	# TO UNCOMMENT
        if self.check_new_state(new_location):
          self.location = new_location
          #self.orientation = new_orientation
          self.successes += 1
          return
      else:
        if self.check_new_state(new_orientation):
          #self.orientation = new_orientation
          self.successes += 1
          return
	
  def time_integrator_AB2AM2(self, dt, current_ite):
    ''' Take a timestep of length dt using the RFD method '''
    # Attempt steps until we find a valid endpoint.  
    #print "self.dim =" , self.dim
    while True:
      if self.has_location:

	#print"############### Predictor: AB2"
        self.rfd_time_step(dt,current_ite)
        new_location = []
        #print "self.location = ",self.location
	#print "self.veltot = ",self.veltot
	#print "self.veltot[0] = ",self.veltot[0]
	#print "self.velocity_previous = ",self.velocity_previous
	
	if current_ite==0:
	  
	  #print "Use Explicit Euler for 1st time Step"
	  for i in range(self.Nrods): 
	    new_location.append(self.location[i] +\
		dt*self.veltot[3*i:3*(i+1)]) 
	else:
	  
	  for i in range(self.Nrods): 
	     #print"# We now use AB2 in the absence of fluctuations"
	    new_location.append(self.location[i] +\
		dt*(1.5*self.veltot[3*i:3*(i+1)] - 0.5*self.velocity_previous[3*i:3*(i+1)]))

          # V^p(n+1)
	  self.velocity_previous = self.veltot
	  # X(n)
	  location_save = self.location
	  # X^p(n+1)
	  self.location = new_location
	    
	  #print"############### Corrector AM2"
	  self.rfd_time_step(dt,current_ite)
	  new_location = []
	  for i in range(self.Nrods): 
	      # We now use AM2 in the absence of fluctuations
	      new_location.append(location_save[i] +\
		  0.5*dt*(self.veltot[3*i:3*(i+1)] + self.velocity_previous[3*i:3*(i+1)]))
	  #print "self.location = ",self.location
	  #print "new_location = ",new_location
	  
        
           ## TO COMMENT, ONLY FOR HYDRO TESTS
          #self.mob_coeff.append(mobility_tt)

          ## TO COMMENT, ONLY FOR HYDRO TESTS
          #new_location.append(self.location[i] + 0.008637333333333*np.array([0.0, 0.0, (-1.0)**(float(i+1))]))
	
     
      self.velocity_previous = self.veltot
      
      # Check validity of new state.
      if self.has_location:
	# TO UNCOMMENT
        if self.check_new_state(new_location):
          self.location = new_location
          #self.orientation = new_orientation
          self.successes += 1
          return
      else:
        if self.check_new_state(new_orientation):
          #self.orientation = new_orientation
          self.successes += 1
          return
		
  def time_integrator_AB1AM2(self, dt, current_ite):
    ''' Take a timestep of length dt using the RFD method '''
    # Attempt steps until we find a valid endpoint.  
    #print "self.dim =" , self.dim
    while True:
      if self.has_location:

	#print"############### Predictor: AB1"
        self.rfd_time_step(dt,current_ite)
        new_location = []
        #print "self.location = ",self.location
	#print "self.veltot = ",self.veltot
	#print "self.veltot[0] = ",self.veltot[0]
	#print "self.velocity_previous = ",self.velocity_previous
	
	if current_ite==0:
	  
	  #print "Use Explicit Euler for 1st time Step"
	  for i in range(self.Nrods): 
	    new_location.append(self.location[i] +\
		dt*self.veltot[3*i:3*(i+1)]) 
	else:
	  
	  for i in range(self.Nrods): 
	     #print"# We now use AB1 in the absence of fluctuations"
	    new_location.append(self.location[i] +\
		dt*self.veltot[3*i:3*(i+1)])

          # V^p(n+1)
	  self.velocity_previous = self.veltot
	  # X(n)
	  location_save = self.location
	  # X^p(n+1)
	  self.location = new_location
	    
	  #print"############### Corrector AM2"
	  self.rfd_time_step(dt,current_ite)
	  new_location = []
	  for i in range(self.Nrods): 
	      # We now use AM2 in the absence of fluctuations
	      new_location.append(location_save[i] +\
		  0.5*dt*(self.veltot[3*i:3*(i+1)] + self.velocity_previous[3*i:3*(i+1)]))

     
      self.velocity_previous = self.veltot
      
      # Check validity of new state.
      if self.has_location:
	# TO UNCOMMENT
        if self.check_new_state(new_location):
          self.location = new_location
          #self.orientation = new_orientation
          self.successes += 1
          return
      else:
        if self.check_new_state(new_orientation):
          #self.orientation = new_orientation
          self.successes += 1
          return
  	
  def time_integrator_EE(self, dt, current_ite):
    ''' Take a timestep of length dt using the RFD method '''
    # Attempt steps until we find a valid endpoint.  
    #print "self.dim =" , self.dim
    while True:
      if self.has_location:

	#print"############### Predictor: AB2"
        self.rfd_time_step(dt,current_ite)
        new_location = []
     

	#print "Use Explicit Euler for 1st time Step"
	for i in range(self.Nrods): 
	  new_location.append(self.location[i] +\
	      dt*self.veltot[3*i:3*(i+1)]) 
	
     
      self.velocity_previous = self.veltot
      
      # Check validity of new state.
      if self.has_location:
	# TO UNCOMMENT
        if self.check_new_state(new_location):
          self.location = new_location
          #self.orientation = new_orientation
          self.successes += 1
          return
      else:
        if self.check_new_state(new_orientation):
          #self.orientation = new_orientation
          self.successes += 1
          return  
	
	
	
  def additive_em_time_step(self, dt):
    ''' 
    Take a simple Euler Maruyama step assuming that the mobility is
    constant.  This for testing and debugging.  We also use it to make sure
    that we need the drift for the correct distribution, etc.
    '''
    if self.has_location:
      mobility  = self.mobility(self.location, self.orientation)
      mobility_half = np.linalg.cholesky(mobility)
      noise = np.random.normal(0.0, 1.0, self.dim*6)
      force = self.force_calculator(self.location, self.orientation)
      torque = self.torque_calculator(self.location, self.orientation)
      velocity_and_omega = (np.dot(mobility, np.concatenate([force, torque])) +
                            np.sqrt(2.0*self.kT/dt)*
                            np.dot(mobility_half, noise))
      velocity = velocity_and_omega[0:(3*self.dim)]
      omega = velocity_and_omega[(3*self.dim):(6*self.dim)]
      new_location = self.location + dt*velocity
      new_orientation = []
      for i in range(self.dim):
        quaternion_dt = Quaternion.from_rotation((omega[(i*3):(i*3+3)])*dt)
        new_orientation.append(quaternion_dt*self.orientation[i])
      if self.check_new_state(new_location, new_orientation):
        self.successes += 1
        self.orientation = new_orientation
        self.location = new_location
    else:
      mobility = self.mobility(self.orientation)
      mobility_half = np.linalg.cholesky(mobility)
      torque = self.torque_calculator(self.orientation)
      noise = np.random.normal(0.0, 1.0, self.dim*3)
      omega = (np.dot(mobility, torque) + 
               np.sqrt(2.0*self.kT/dt)*np.dot(mobility_half, noise))
      new_orientation = []
      for i in range(self.dim):
        quaternion_dt = Quaternion.from_rotation((omega[(i*3):(i*3+3)])*dt)
        new_orientation.append(quaternion_dt*self.orientation[i])

      if self.check_new_state(None, new_orientation):
        self.successes += 1
        self.orientation = new_orientation

  def estimate_divergence(self):
    ''' 
    Estimate the divergence term with a deterministic
    finite difference approach. This is for a single quaternion.
    '''
    delta = 1e-6
    div_term = np.zeros(3)
    for k in range(3):
      omega = np.zeros(3)
      omega[k] = 1.
      quaternion_dt = Quaternion.from_rotation(omega*delta/2.)
      quaternion_tilde_1 = quaternion_dt*self.orientation[0]
      quaternion_dt = Quaternion.from_rotation(-1.*omega*delta/2.)
      quaternion_tilde_2 = quaternion_dt*self.orientation[0]
      div_term += np.inner((self.mobility([quaternion_tilde_1]) -
                            self.mobility([quaternion_tilde_2])),
                           omega/delta)
      
    return div_term

    
  def estimate_drift_and_covariance(self, dt, n_steps, scheme):
    ''' Emperically estimate the drift and covariance term in the absence of torque. 
    For now this is just without location.  TODO: add location.'''

    if self.dim > 1:
      # For now, hard code to 1 dimensional integrator.
      raise NotImplementedError('Drift and Covariance estimation only implemented for '
                                '1-d integrators.')

    old_torque = self.torque_calculator
    if self.has_location:
      def zero_torque(orientation, location):
        return np.zeros(3*len(orientation))
      def zero_force(orientation, location):
        return np.zeros(3*len(orientation))
      old_force = self.force_calculator
      self.force_calculator = zero_force
      initial_location = self.location
    else:
      def zero_torque(orientation):
        return np.zeros(3*len(orientation))
    self.torque_calculator = zero_torque
    initial_orientation = self.orientation

    drift_samples = []
    covariance_samples = []
    for k in range(n_steps):
      if scheme == 'FIXMAN':
        self.fixman_time_step(dt)
      elif scheme == 'RFD':
        self.rfd_time_step(dt)
      else:
        raise Exception('scheme must be FIXMAN or RFD for drift estimation.')

      for l in range(self.dim):
        orientation_increment = self.orientation[l]*initial_orientation[l].inverse()
        drift = orientation_increment.rotation_angle()
        if self.has_location:
          drift = np.concatenate([self.location[l] - initial_location[l], drift])
      drift_samples.append(drift)
      covariance_samples.append(np.outer(drift, drift))
      self.orientation = initial_orientation
      if self.has_location:
        self.location = initial_location

    avg_drift = np.mean(drift_samples, axis=0)/dt
    avg_covariance = np.mean(covariance_samples, axis=0)/(2.*dt)

    # Reset torque calculator
    self.torque_calculator = old_torque
    if self.has_location:
      self.force_calculator = old_force

    return [avg_drift, avg_covariance]
      
      
  def check_new_state(self, location):
    ''' 
    Use the check function to test if the new state is valid.
    If not, the timestep will be thrown out. 
    '''
    if self.check_function:
      if self.has_location:
        admissible = self.check_function(location)
      else:
        admissible = self.check_function(location)
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
    