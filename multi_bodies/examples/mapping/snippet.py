'''
Code to save the bodies velocities and the flow in a rectangular grid and a spherical Chebyshev-Fourier grid.
It can be copy-pasted to quaternion_integrator_multi_bodies.py after the solution of the linear mobility problem has been solved.
For example, at line 1585.
'''

      if True:
        # Save bodies velocity
        mode = 'w' if step == 0 else 'a'
        name = self.output_name + '.bodies_velocities.dat'
        with open(name, mode) as f_handle:
          np.savetxt(f_handle, velocities.reshape((len(self.bodies), 6)))
        
        # Set radius of blobs 
        radius_source = np.zeros(self.Nblobs)
        offset = 0
        for b in self.bodies:
          num_blobs = b.Nblobs
          radius_source[offset:(offset+num_blobs)] = b.blobs_radius
          offset += num_blobs

        # Extract blob forces 
        lambda_blobs = sol_precond[0 : 3*self.Nblobs]

        # Get blobs vectors
        r_vectors_blobs = self.get_blobs_r_vectors(self.bodies, self.Nblobs)
        
        # Plot flow in rectangular in the body frame of reference of body zero 
        if self.plot_velocity_field.size > 0:
          pvf.plot_velocity_field(self.plot_velocity_field,
                                  r_vectors_blobs,
                                  lambda_blobs,
                                  self.a,
                                  self.eta,
                                  self.output_name + '.step.' + str(step).zfill(8),
                                  0,
                                  radius_source=radius_source,
                                  frame_body = self.bodies[0],
                                  mobility_vector_prod_implementation='numba_no_wall')

        # Plot flow on a shell
        if True:
          sphere_radius = 16
          p = 32
          self.plot_velocity_field_shell(r_vectors_blobs,
                                         lambda_blobs,
                                         self.a,
                                         self.eta,
                                         sphere_radius,
                                         p,
                                         self.output_name + '.step.'+str(step).zfill(8) + '.sphere_radius.'+str(sphere_radius) + '.p.'+ str(p) + '.velocity_field_sphere.dat',
                                         frame_body=self.bodies[0],
                                         mobility_vector_prod_implementation='numba_no_wall')



          
