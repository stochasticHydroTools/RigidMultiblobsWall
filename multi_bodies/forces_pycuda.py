'''
This module compute the blob-blob forces using pycuda.
'''

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# These lines set the precision of the cuda code
# to single or double. Set the precision
# in the following lines and edit the lines
# after   'mod = SourceModule("""'    accordingly
precision = 'single'
# precision = 'double'

mod = SourceModule("""
// Set real to single or double precision.
// This value has to agree witht the value
// for precision setted above.
typedef float real;
// typedef double real;

#include <stdio.h>
/*
  This function compute the force between two blobs
  with vector between blob centers r.

  In this example the force is derived from the potential
  
  U(z) = U0 + U0 * (a-z)/b   if z<a
  U(z) = U0 * exp(-(z-a)/b)  iz z>=a
  
  with
  eps = potential strength
  r_norm = distance between blobs
  b = Debye length
  a = blob_radius
*/
__device__ void blob_blob_force(const real rx, 
                                const real ry, 
                                const real rz, 
                                real &fx, 
                                real &fy, 
                                real &fz, 
                                const real eps, 
                                const real b,
                                const real a){

  real r = sqrt(rx*rx + ry*ry + rz*rz);
  real f;
  if(r > 2*a){
    f = -(eps / b) * exp(-(r-2*a) / b) / r; 
  }
  else if(r > 0){
    f = -(eps / b) / r;
  }
  fx += f * rx;
  fy += f * ry;
  fz += f * rz;
}

/*
 This function computes the blob-blob force for all blobs.
*/
__global__ void calc_blob_blob_force(const real *x, 
                                     real *f, 
                                     const real repulsion_strength, 
                                     const real debye_length,
                                     const real blob_radius,
                                     const real Lx,
                                     const real Ly,
                                     const real Lz,
                                     const int number_of_blobs){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;   

  int offset_i = i * 3;
  int offset_j;
  real rx, ry, rz;
  real fx = 0;
  real fy = 0;
  real fz = 0;

  // Loop over blobs to add interanctions
  for(int j=0; j<number_of_blobs; j++){
    offset_j = j * 3;

    // Compute blob to blob vector
    rx = x[offset_j]     - x[offset_i];
    ry = x[offset_j + 1] - x[offset_i + 1];
    rz = x[offset_j + 2] - x[offset_i + 2];

    // Project a vector r to the minimal image representation
    // centered around (0,0,0) and of size L=(Lx, Ly, Lz). If 
    // any dimension of L is equal or smaller than zero the 
    // box is assumed to be infinite in that direction.
    if(Lx > 0){
      rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
    }
    if(Ly > 0){
      ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
    }
    if(Lz > 0){
      rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
    }

    // Compute force between blobs i and j
    if(i != j){
      blob_blob_force(rx, ry, rz, fx, fy, fz, repulsion_strength, debye_length, blob_radius);
    }
  }
  
  // Return forces
  f[offset_i]     = fx;
  f[offset_i + 1] = fy;
  f[offset_i + 2] = fz;
}
""")

def real(x):
  if precision == 'single':
    return np.float32(x)
  else:
    return np.float64(x)


def set_number_of_threads_and_blocks(num_elements):
  '''
  This functions uses a heuristic method to determine
  the number of blocks and threads per block to be
  used in CUDA kernels.
  '''
  threads_per_block=512
  if((num_elements//threads_per_block) < 512):
    threads_per_block = 256
  if((num_elements//threads_per_block) < 256):
    threads_per_block = 128
  if((num_elements//threads_per_block) < 128):
    threads_per_block = 64
  if((num_elements//threads_per_block) < 128):
    threads_per_block = 32
  num_blocks = (num_elements-1)//threads_per_block + 1

  return (threads_per_block, int(num_blocks))


def calc_blob_blob_forces_pycuda(r_vectors, *args, **kwargs):
   
  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length')
  eps = kwargs.get('repulsion_strength')
  b = kwargs.get('debye_length')
  blob_radius = kwargs.get('blob_radius')

  # Reshape arrays
  x = real(np.reshape(r_vectors, number_of_blobs * 3))
  f = real(np.empty_like(x))
        
  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  f_gpu = cuda.mem_alloc(f.nbytes)
    
  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
    
  # Get blob-blob force function
  force = mod.get_function("calc_blob_blob_force")

  # Compute mobility force product
  force(x_gpu, f_gpu, real(eps), real(b), real(blob_radius), real(L[0]), real(L[1]), real(L[2]), number_of_blobs, block=(threads_per_block, 1, 1), grid=(num_blocks, 1)) 

  # Copy data from GPU to CPU (device to host)
  cuda.memcpy_dtoh(f, f_gpu)

  return np.reshape(f, (number_of_blobs, 3))
