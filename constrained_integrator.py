''' Constrained Integrator object used to integrate langevin equations 
on a given constraint.  Uses a projection method to stay on the constraint, and
an RFD term to generate the correct (slightly modified) thermal drift to account for the
surface measure of the constraint. '''

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
    # Set random generator.
    self.random_generator = np.random.normal
    #TODO: make this dynamic somehow.
    self.rfdelta = 1.0e-6
    self.surface_function = surface_function
    self.mobility = mobility
    self.dim = self.mobility(initial_position).shape[0]
    if mobility(initial_position).shape[1] != self.dim:
      print 'Mobility Matrix must be square.  # rows is ', self.dim, 
      print ' # Columns is ', self.mobility(initial_position).shape[1]
      sys.exit()

    self.current_time = 0.

    # Initial Position must be a 'matrix.'
    if (initial_position.shape[0] == self.dim and 
        self.surface_function(initial_position) == 0.):
      self.position = initial_position
      # Save initial position so we can reset for multiple runs.
      self.initial_position = initial_position
    else:
      print 'initial position.shape is ', initial_position.shape
      print ('constraint function of initial position is ', 
             self.surface_function(initial_position))
      raise ValueError('Initial position is either the wrong dimension or'
                        ' is not on the constraint.')
      
    if scheme not in ['RFD', 'OTTINGER']:
      print 'Only RFD and Ottinger Schemes are implemented'
      raise NotImplementedError('Only RFD and Ottinger schemes are implemented')
    else:
      self.scheme = scheme

    self.path = [self.position]


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
    
    self.ProjectToConstraint()

    self.SavePath(self.position)

        
  def OttingerTimeStep(self, dt):
    ''' Take a step of the Ottinger scheme '''
    raise NotImplementedError('Ottinger Scheme not yet Implemented.')


  def RFDTimeStep(self, dt):
    ''' Take a step of the RFD scheme '''
    #TODO: Make this variable
    kT = 1.0

    w_tilde = np.matrix([[a] for a in self.random_generator(0.0, 1.0, self.dim)])
    w = np.matrix([[a] for a in self.random_generator(0.0, 1.0, self.dim)])
    p_l2 = self.ProjectionMatrix(self.position, np.matrix(np.eye(2,2)))
    predictor_position = self.position + self.rfdelta*p_l2*w_tilde
    # For now we have no potential.
    # force = np.matrix([[0.] for _ in range(self.dim)])
    p = self.ProjectionMatrix(self.position)
    p_tilde = self.ProjectionMatrix(predictor_position)
    mobility = self.mobility(self.position)
    noise_magnitude = self.NoiseMagnitude(self.position)
    mobility_tilde = self.mobility(predictor_position)
    # (self.position + dt*p*mobility*force +
    corrector_position = self.position + ((dt*kT/self.rfdelta)*(p_tilde*mobility_tilde
                                                - p*mobility)*p_l2*w_tilde +
                          np.sqrt(2*kT*dt)*p*noise_magnitude*w)

    self.position = corrector_position



  def SavePath(self, position):
    self.path.append(position)

  def ResetPath(self):
    ''' 
    Clear the path variable for a new run, and set 
    position to initial_position 
    '''
    self.position = self.initial_position
    self.path = [self.position]

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
    vector_magnitude = 0.0
    for k in range(self.dim):
      direction = np.matrix([[0.0] for _ in range(self.dim)])
      direction[k,0] = self.rfdelta
      normal_vector[k,0] = ((self.surface_function(position + direction) - 
                          self.surface_function(position - direction))/
                          (2.0*self.rfdelta))
      vector_magnitude += normal_vector[k,0]**2
    
    vector_magnitude = np.sqrt(vector_magnitude)
    for k in range(self.dim):
        normal_vector[k,0] /= vector_magnitude
    return normal_vector

      
  def ProjectionMatrix(self, position, D_matrix = None):
    ''' Calculate projection matrix at given position,
    P_m = I - (Dn X n) /(n^tDn) 
      args
        position:  dim x 1 vector of floats - position to evaluate n at
                      (and later mobility).
        D_matrix:  dim x dim matrix of floats - D in the above expression for
                     the projection matrix.
    '''
    if D_matrix is None:
      D_matrix = self.mobility(position)
    normal_vector = self.NormalVector(position)

    # First calcualate n^t M n for denominator.
    nMn = normal_vector.T*D_matrix*normal_vector

    # Now calculate projection matrix.
    projection = np.matrix([np.zeros(self.dim) for _ in range(self.dim)])
    for j in range(self.dim):
      for k in range(self.dim):
        projection[j,k] = -1.*((D_matrix[j]*normal_vector)*
                               normal_vector[k,0])
        projection[j,k] /= nMn
      projection[j,j] += 1.0

    return projection


  def NoiseMagnitude(self, position):
    ''' 
    Calculate cholesky decomposition of mobility for noise term.  For
    now this just works on diagonal matrices.  
    args 
      position:  np.matrix - position where we evaluate M^1/2, 
                            the noise magnitude.
    returns
      noise_magnitude: np.matrix - Square root of mobility evaluated 
                                   at position.
    NOTE: FOR NOW THIS IS ONLY IMPLEMENTED FOR DIAGONAL MOBILITY.
    '''
    noise_magnitude = np.matrix([np.zeros(self.dim) for _ in range(self.dim)])
    mobility_matrix = self.mobility(position)
    for j in range(self.dim):
      for k in range(self.dim):
        if j == k:
          noise_magnitude[j, k] = np.sqrt(mobility_matrix[j, k])
        elif mobility_matrix[j, k] != 0:
          raise NotImplementedError('Noise magnitude for non-diagonal'
                                    'mobility not yet implemented')

    return noise_magnitude

  def ProjectToConstraint(self):
    ''' Project the current position to the nearest point on
    the constraint with a newtons method line search.  We search
    along the gradient of the constraint function, so we have:
      0 \approx q(x) + \grad(q) \cdot \grad(q) \alpha
      => \alpha = -q(x)/||\grad(q)||^2
      x_new = x + \grad_q*\alpha
    we then iterate until we get the constraint close to 0.
    '''
    TOL = 1e-5
    iteration_num = 0
    while (np.abs(self.surface_function(self.position)) > TOL and
           iteration_num <= 5):
      grad_q = np.matrix([[0.] for _ in range(self.dim)])
      vector_magnitude = 0.0
      for k in range(self.dim):
        direction = np.matrix([[0.0] for _ in range(self.dim)])
        direction[k,0] = self.rfdelta
        grad_q[k,0] = ((self.surface_function(self.position + direction) - 
                        self.surface_function(self.position - direction))/
                       (2.0*self.rfdelta))
        vector_magnitude += grad_q[k,0]**2
        
      vector_magnitude = np.sqrt(vector_magnitude)
        
      # Line search alpha
      alpha = -1.*self.surface_function(self.position)/(vector_magnitude**2)
      self.position = self.position + alpha*grad_q
      iteration_num += 1

    if iteration_num == 5:
      print ('WARNING: 5 iterations of line search without getting within ' +
            'tolerance of the constraint. TOL = ', TOL)
      
      
