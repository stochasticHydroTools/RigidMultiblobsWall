'''
Simple integrator for N quaternions.
'''


class QuaternionIntegrator(object):
  '''
  Integrator that timesteps using Fixman quaternion updates.
  '''
  def __init__(self, mobility, initial_position, ):
    '''
    Set up components of the integrator.  
    args: 
    
      mobility: function that takes a vector of torques and positions
                (quaternions), and calculates the angular velocity on each
                quaternion at that location.

      initial_position: vector of quaternions representing the initial configuration
                        of the system.
    

    '''
    self._mobility = mobility
    self._initial_position = initial_position
    
  def TimeStep(self, dt):
    pass
    
    
