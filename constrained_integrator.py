import sys
import numpy as np

class ConstrainedIntegrator(object):
  '''A class intended to test out temporal integration schemes
  for constrained diffusion. '''

  def __init__(self, surface_function, mobility, scheme, initial_position):
    ''' Initialize  the integrator object, it needs a surface parameterization,
    a timestepping scheme, and a Mobility Matrix.

    args
      scheme:            string - must be one of 'OTTINGER' or 'RFD'.  OTTINGER indicates
                           an unconstrained step that is projected back to the surface.
                           RFD gives a projected step using the Random Finite Difference to
                           generate the drift.
      surface_function:  function - takes the coordinates and returns the value of 
                           the constraint function.  
                           The constraint is such that surface_function(x) = 0.
      mobility:          np.matrix of floats - mobility matrix.  Stored as an array of arrays
                           with the first index representing the row. Must be a square matrix.
      initial_position:  np.matrix of floats - initial position of the system.  Must be 1 X 
                           the same dimension as the mobility matrix, and we must have
                           surface_function(initial_position) = 0.
    '''
    self.random_generator = np.random.normal
    #TODO: make this dynamic somehow.
    self.rfdelta = 1.0e-8
    self.surface_function = surface_function
    #TODO: make this a function of position that returns a matrix.
    self.mobility = mobility
    self.dim = mobility.shape[0]
    if mobility.shape[1] != self.dim:
      print 'Mobility Matrix must be square.  # rows is ', self.dim, 
      print ' # Columns is ', len(self.mobility[k])
      sys.exit()

    self.current_time = 0.

    # Initial Position must be a matrix.
    if (initial_position.shape[0] == self.dim and 
        self.surface_function(initial_position) == 0.):
      self.position = initial_position
    else:
      print "initial position.shape[0] is ", initial_position.shape[0]
      raise ValueError('Initial position is either the wrong dimension or'
                        ' is not on the constraint.')
      
    if scheme not in ['RFD', 'OTTINGER']:
      print 'Only RFD and Ottinger Schemes are implemented'
      raise NotImplementedError('Only RFD and Ottinger schemes are implemented')
    else:
      self.scheme = scheme


  def MockRandomGenerator(self):
    ''' For testing, replace random generator with something that just returns 1. '''
    def OnlyOnesRandomGenerator(a, b, n):
      return np.ones(n)
    self.random_generator = OnlyOnesRandomGenerator

    
  def TimeStep(self, dt):
    ''' Step from current time to next time with timestep of size dt.
     args
       dt: float - time step size.
     '''
    if self.scheme == 'RFD':
      self.RFDTimeStep(dt)
    elif self.scheme == 'OTTINGER':
      self.OttingerTimeStep(dt)
    else:
      print 'Should not get here in TimeStep.'
      sys.exit()

        
  def OttingerTimeStep(self, dt):
    ''' Take a step of the Ottinger scheme '''
    raise NotImplementedError('Ottinger Scheme not yet Implemented.')


  def RFDTimeStep(self, dt):
    ''' Take a step of the RFD scheme '''
    #TODO: Make this dynamic
    kT = 1.0
    w_tilde = np.matrix([[a] for a in self.random_generator(0.0, 1.0, self.dim)])
    print "w_tilde is ", w_tilde
    w = np.matrix([[a] for a in self.random_generator(0.0, 1.0, self.dim)])
    print 'w is ', w
    predictor_position = self.position + self.rfdelta*w_tilde

    print 'predictor_position is ', predictor_position
      
    # For now we have no potential.
    force = np.matrix([[0.] for _ in range(self.dim)])
    p = self.ProjectionMatrix(self.position)
    p_tilde = self.ProjectionMatrix(predictor_position)
    print 'p is ', p
    print 'p_tilde is ', p_tilde
    print "RFD drift is ", (dt*kT/self.rfdelta)*(p_tilde - p)*w_tilde
    print "stochastic term is ", np.sqrt(2*kT*dt)*p*self.mobility*w
    #TODO: variable mobility and mobility factor.
    #TODO: This is incorrect, I need an L2 projection for drift.
    corrector_position = (self.position + dt*p*self.mobility*force +
                          (dt*kT/self.rfdelta)*(p_tilde*w_tilde - p*w_tilde) +
                          np.sqrt(2*kT*dt)*p*self.mobility*w)

    self.position = corrector_position


  def NormalVector(self, position):
    ''' At the given position, calculate normalized gradient
    of the surface_function numerically. 
    
    args
      position:  vector (1 X self.dim matrix) of floats - position to evaluate normal vector.

    output
      normal_vector: array of floats - normalized gradient of the surface 
                       funtion at self.position, evaluated numerically.
    '''
    normal_vector = np.matrix([[0.] for _ in range(self.dim)])
    delta = 1.0e-8
    vector_size = 0.0
    for k in range(self.dim):
      direction = np.matrix([[0.0] for _ in range(self.dim)])
      direction[k,0] = delta
      normal_vector[k,0] = ((self.surface_function(position + direction) - 
                          self.surface_function(position - direction))/
                          (2.0*delta))
      vector_size += normal_vector[k,0]**2
    
    vector_size = np.sqrt(vector_size)
    for k in range(self.dim):
        normal_vector[k,0] /= vector_size
      
    print "normal vector is ", normal_vector
    return normal_vector

      
  def ProjectionMatrix(self, position):
    ''' Calculate projection matrix at given position,
    P_m = delta_ik - (Mn X n) /(n^tMn) '''
    normal_vector = self.NormalVector(position)
    projection = np.matrix([np.zeros(self.dim) for _ in range(self.dim)])

    # First calcualate n^t M n for denominator.
    nMn = normal_vector.T*self.mobility*normal_vector

    # Now calculate projection matrix.
    projection = np.matrix([np.zeros(self.dim) for _ in range(self.dim)])
    for j in range(self.dim):
      for k in range(self.dim):
        projection[j,k] = -1.*((self.mobility[j]*normal_vector)*
                               normal_vector[k,0])
        projection[j,k] /= nMn
      projection[j,j] += 1.0

    return projection

  
        
    
