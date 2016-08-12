import numpy as np
import sys
from body import body 


def slip_extensile_rod(body):
  '''
  Creates slip for a extensile rod. The slip is along the
  axis pointing to the closest end. We assume the blobs
  are equispaced.
  In this version the slip is constant. 
  '''

  # Slip speed
  speed = -20.0
  if body.Nblobs == 14:
   Nblobs_covering_ends = 0
   Nlobs_perimeter = 0
  elif body.Nblobs == 86:
   Nblobs_covering_ends = 1
   Nlobs_perimeter = 6 
  elif body.Nblobs == 324:
   Nblobs_covering_ends = 6
   Nlobs_perimeter = 12 
  
  slip = np.empty((body.Nblobs, 3))

  # Get rod orientation
  r_vectors = body.get_r_vectors()
    
  if body.Nblobs>14:
    axis = r_vectors[body.Nblobs- 2*Nblobs_covering_ends-2] - r_vectors[Nlobs_perimeter-2]
  else:
    axis = r_vectors[body.Nblobs-1] - r_vectors[0]

  length_rod = np.linalg.norm(axis)+2.0*body.blob_radius
  axis = axis / np.sqrt(np.dot(axis, axis))
  
  distance_from_center = 0.8
  lower_bound = length_rod/2.0 - distance_from_center   
  upper_bound = length_rod/2.0

  # Create slip  
  slip_blob = []
  for i in range(body.Nblobs):
		if (Nblobs_covering_ends>0) and (i>=body.Nblobs-2*Nblobs_covering_ends):
			slip_blob = [0., 0., 0.]
		else:
			dist_COM_along_axis = np.dot(r_vectors[i]-body.location, axis)
			if (dist_COM_along_axis >lower_bound) and (dist_COM_along_axis <=upper_bound):
				slip_blob = -speed * axis
			elif (dist_COM_along_axis <-lower_bound) and (dist_COM_along_axis >=-upper_bound):
				slip_blob = speed * axis
			else:
				slip_blob = [0., 0., 0.]
	  		   
		slip[i] = slip_blob
      

  return slip

