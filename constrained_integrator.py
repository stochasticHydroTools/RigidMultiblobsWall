

class ConstrainedIntegrator(object):
  """A class intended to test out temporal integration schemes
  for constrained diffusion. """

  def __init__(self, surface_function, mobility):
    """ Initialize  the integrator object, it needs a surface parameterization
    and a Mobility Matrix."""
    self.surface_function = surface_function
    self.mobility = mobility
    self.current_time = 0.
    
  def TimeStep(self, scheme, dt):
    """ Step from current time to next time with timestep of size dt.

     args
        scheme:  string - must be one of "EULER" or "RFD".  EULER indicates
                   an unconstrained step that is projected back to the surface.
                   RFD gives a projected step using the Random Finite Difference to
                   generate the drift.
        dt:  float - time step size.
     """
    
    
        
        
