




# Parameters
A = 0.2625  # Radius of blobs in uM
ETA = #







def get_boomerang_r_vectors(location, orientation):
    '''Get the vectors of the 7 blobs used to discretize the boomerang.

          1 2 3 4    
    
          O-O-O-O
                O 5
                O 6
                O 7
   
    The location is the location of blob 4.  Initial configuration is in the
    x-y plane, with  arm 1-2-3  pointing in the positive x direction, and arm
    5-6-7 pointing in the positive y direction.
    Seperation between blobs is currently hard coded at 
    '''
    
    initial_r1 = np.array([]
