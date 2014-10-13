

class ConstrainedIntegrator(object):
  """A class intended to test out temporal integration schemes
  for constrained diffusion. """

  def __init__(self, surface_function, mobility):
    """ Initialize  the integrator object, it needs a surface parameterization
    and a Mobility Matrix."""
    self.surface_function = surface_function
    self.mobility = mobility

  def TimeStep(self, scheme):
    """ 
    
        
        
