
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


mod = SourceModule("""
#include <stdio.h>

/*
  Cumpute the enery coming from one blob potentials,
  e.g. gravity or interactions with the wall.

  In this example we add gravity and a repulsion with the wall;
  the interaction with the wall is derived from a Yukawa-like
  potential
  U = e * a * exp(-(h-a) / b) / (h - a)
  with 
  e = repulsion_strength_wall
  a = blob_radius
  h = distance to the wall
  b = debye_length_wall

*/
__device__ void one_blob_potential(double &u, 
                                   const double rx, 
                                   const double ry, 
                                   const double rz, 
                                   const double blob_radius, 
                                   const double debye_length_wall, 
                                   const double eps_wall, 
                                   const double weight){
  // Add gravity
  u += weight * rz;

  // Add interaction with the wall
  u += eps_wall * blob_radius * exp(-(rz-blob_radius) / debye_length_wall) / abs(rz - blob_radius);  

  // If blob overlaps with the wall increase the energy by a large magnitude;
  // this prevents the system to access forbidden configurations.
  if(rz < blob_radius){
    u += eps_wall * 1e+12;
  }
}

/*
  Compute the energy coming from blob-blob potentials.
  In this example the force is derived from a Yukawa potential
  
  U = eps * exp(-r_norm / b) / r_norm
  
  with
  eps = potential strength
  r_norm = distance between blobs
  b = Debye length
*/
__device__ void blob_blob_potential(double &u,
                                    const double rx,
                                    const double ry,
                                    const double rz,
                                    const int i,
                                    const int j,
                                    const double debye_length,
                                    const double eps,
                                    const double blob_radius){                
  if(i != j){
    double r = sqrt(rx*rx + ry*ry + rz*rz);
    u += eps * exp(-r / debye_length) / r;
  }
}

/*
  Compute blobs energy. It takes into account both
  single blob and two blobs contributions.
*/
__global__ void potential_from_position_blobs(const double *x,
                                              double *total_U, 
                                              const int n_blobs,
                                              const double Lx,
                                              const double Ly,
                                              const double debye_length_wall,
                                              const double eps_wall,
                                              const double debye_length,
                                              const double eps,
                                              const double weight,
                                              const double blob_radius){

  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= n_blobs) return;   

  double u = 0.0;
  double rx, ry, rz;
  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  if (x[ioffset+2] > 0){
    // 1. One blob potential
    one_blob_potential(u, x[ioffset], x[ioffset+1], x[ioffset+2], blob_radius, debye_length_wall, eps_wall, weight);

    // 2. Two blobs potential
    // IMPORTANT, we don't want to compute the blob-blob interaction twice! 
    // See the loop limits.
    for(int j=i+1; j<n_blobs; j++){
      joffset = j * NDIM;
      // Compute vector between particles i and j    
      rx = x[ioffset    ] - x[joffset    ];
      ry = x[ioffset + 1] - x[joffset + 1];
      rz = x[ioffset + 2] - x[joffset + 2];
      if (Lx > 0){
        rx = rx - int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx;
      }
      if (Ly > 0){
        ry = ry - int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly;
      }
      // Compute blob-blob interaction
      blob_blob_potential(u, rx, ry, rz, i, j, debye_length, eps, blob_radius);
    }
  }
  else
  {
    // make u large for blobs behind the wall
    // if a particle starts out of bounds somehow, then it won't want to move further out
    u = 1e+05*(-x[ioffset+2] +1); 
  }
  //IF END
  //3. Save potential U_i
  total_U[i] = u;
  return;
}

/*
  Compute one body potentials.
  Default is zero.
*/
__device__ void one_body_potential(double &u, 
                                   const double rx, 
                                   const double ry, 
                                   const double rz,
                                   const double q1,
                                   const double q2,
                                   const double q3,
                                   const double q4){
  return;
}

/*
  Compute body-body potentials. Default is zero.
*/
__device__ void body_body_potential(double &u, 
                                    const double rx, 
                                    const double ry,
                                    const double rz,
                                    const double q1i,
                                    const double q2i,
                                    const double q3i,
                                    const double q4i,
                                    const double q1j,
                                    const double q2j,
                                    const double q3j,
                                    const double q4j,
                                    const int i, 
                                    const int j){
  
  return;
}


/*
  Compute bodies energy. It takes into account both
  single body and two bodies contributions.
*/
__global__ void potential_from_position_bodies(const double *x,
                                               const double *q,
                                               double *total_U, 
                                               const int n_bodies,
                                               const double Lx,
                                               const double Ly){

  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= n_bodies) return;   

  double u = 0.0;
  double rx, ry, rz;
  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  {
    // 1. One body potential
    one_body_potential(u, x[ioffset], x[ioffset+1], x[ioffset+2],
                       q[i*4], q[i*4+1], q[i*4+2], q[i*4+3]);

    // 2. Two blobs potential
    // IMPORTANT, we don't want to compute the body-body interaction twice! 
    // See the loop limits.
    for(int j=i+1; j<n_bodies; j++){
      joffset = j * NDIM;
      // Compute vector between particles i and j    
      rx = x[ioffset    ] - x[joffset    ];
      ry = x[ioffset + 1] - x[joffset + 1];
      rz = x[ioffset + 2] - x[joffset + 2];
      if (Lx > 0){
        rx = rx - int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx;
      }
      if (Ly > 0){
        ry = ry - int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly;
      }
      // Compute blob-blob interaction
      body_body_potential(u, rx, ry, rz, 
                          q[i*4], q[i*4+1], q[i*4+2], q[i*4+3],
                          q[j*4], q[j*4+1], q[j*4+2], q[j*4+3],
                          i, j);
    }
  }

  //IF END
  //3. Save potential U_i
  total_U[i] = u;
  return;
}
""")


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


def blobs_potential(r_vectors, *args, **kwargs):
  '''
  This function compute the energy of the blobs.
  '''
   
  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  periodic_length = kwargs.get('periodic_length')
  debye_length_wall = kwargs.get('debye_length_wall')
  eps_wall = kwargs.get('repulsion_strength_wall')
  debye_length = kwargs.get('debye_length')
  eps = kwargs.get('repulsion_strength')
  weight = kwargs.get('weight')
  blob_radius = kwargs.get('blob_radius')  

  # Reshape arrays
  x = np.reshape(r_vectors, number_of_blobs * 3)
        
  # Allocate CPU memory
  U = np.empty(number_of_blobs)

  # Allocate GPU memory
  utype = np.float64(1.)
  x_gpu = cuda.mem_alloc(x.nbytes)
  u_gpu = cuda.mem_alloc(U.nbytes)
    
  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
    
  # Get pair interaction function
  potential_from_position_blobs = mod.get_function("potential_from_position_blobs")

  # Compute pair interactions
  potential_from_position_blobs(x_gpu, u_gpu,
                                number_of_blobs,
                                np.float64(periodic_length[0]),
                                np.float64(periodic_length[1]),
                                np.float64(debye_length_wall),
                                np.float64(eps_wall),
                                np.float64(debye_length),
                                np.float64(eps),
                                np.float64(weight),
                                np.float64(blob_radius),
                                block=(threads_per_block, 1, 1),
                                grid=(num_blocks, 1)) 
    
  # Copy data from GPU to CPU (device to host)
  cuda.memcpy_dtoh(U, u_gpu)
  return np.sum(U)
  

def bodies_potential(bodies, *args, **kwargs):
  '''
  This function compute the energy of the bodies.
  '''
   
  # Determine number of threads and blocks for the GPU
  number_of_bodies = np.int32(len(bodies))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_bodies)

  # Get parameters from arguments
  periodic_length = kwargs.get('periodic_length')

  # Create location and orientation arrays
  x = np.empty(3 * number_of_bodies)
  q = np.empty(4 * number_of_bodies)
  for k, b in enumerate(bodies):
    x[k*3 : (k+1)*3] = b.location_new
    q[k*4] = b.orientation_new.s
    q[k*4 + 1 : k*4 + 4] = b.orientation_new.p
    
  # Allocate CPU memory
  U = np.empty(number_of_bodies)

  # Allocate GPU memory
  utype = np.float64(1.)
  x_gpu = cuda.mem_alloc(x.nbytes)
  q_gpu = cuda.mem_alloc(q.nbytes)
  u_gpu = cuda.mem_alloc(U.nbytes)
    
  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(q_gpu, q)
    
  # Get pair interaction function
  potential_from_position_bodies = mod.get_function("potential_from_position_bodies")

  # Compute pair interactions
  potential_from_position_bodies(x_gpu, q_gpu, u_gpu,
                                 number_of_bodies,
                                 np.float64(periodic_length[0]),
                                 np.float64(periodic_length[1]),
                                 block=(threads_per_block, 1, 1),
                                 grid=(num_blocks, 1)) 
    
  # Copy data from GPU to CPU (device to host)
  cuda.memcpy_dtoh(U, u_gpu)
  return np.sum(U)


def compute_total_energy(bodies, r_vectors, *args, **kwargs):
  '''
  It computes and returns the total energy of the system as
  
  U = U_blobs + U_bodies
  '''

  # Compute energy blobs
  u_blobs = blobs_potential(r_vectors, *args, **kwargs)

  # Compute energy bodies
  u_bodies = bodies_potential(bodies, *args, **kwargs)

  # Compute and return total energy
  return u_blobs + u_bodies
