import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


mod = SourceModule("""
////////// Wall repulsion //////////////////////////////////////////////////
__device__ void repulsionWall(double h,
                  double a,
                  double &u, 
                  double strength_wall,
                  double invDebyeWallGPU){
  
  u += strength_wall * exp(-h * invDebyeWallGPU) / h;
 
  return;
}
////////// Part-part repulsion //////////////////////////////////////////////////
__device__ void repulsionPart(double rx,
                  double ry,
                  double rz,
                  double &u,
                  int i,
                  int j,
                  double debye_part,
                  double kbT,
                  double epsilon,
                  double diam){
                
  if(i != j){
    double r = sqrt(rx*rx + ry*ry + rz*rz);
    //double invR = 1.0 / r;
    //double r_diam = r - diam;
    //if(r_diam < 0)
      //  u += 100;
    //else
    //Yukawa potential used because it is soft
        //u += 2*kbT * diam * invR * exp(-debye_part * r_diam * diam);
    u += epsilon * exp(-r / debye_part) / r;
    return;
  }
}
//////////harmonic spring at crosspoints///////////////////
__device__ void cross_point_spring_potential(double dx,
                                        double dy,
                                        double dz,
                                        double &u,
                                        double k,
                                        double r_equilibrium){
  double r = sqrt(dx*dx + dy*dy + dz*dz);
  double r2 = (r- r_equilibrium) * (r - r_equilibrium);
  u += 0.5 * k * r2;
}
/*
 force_from_position computes paiwise and wall interactions
*/
__global__ void potential_from_position(const double *x,
                                        double *total_U,                  
                    int n_blobs,
                    double Lx,
                    double Ly,
                    double debye_wall,
                    double strength_wall,
                    double debye_part,
                    double weight,
                    double kbT,
                    double epsilon,
                    int cross1_index,
                    int cross2_index,
                    double diam,
                    double a){

  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= n_blobs) return;   
  double invDebyeWallGPU = 1.0 / debye_wall;
  double Lx_over_2 =  Lx/2.0;
  double Ly_over_2 =  Ly/2.0;
  double u = 0.0;
  double rx, ry, rz;
  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  if (x[ioffset+2]>a){
    // 1. Compute repulsion from the wall
    repulsionWall(x[ioffset+2]-a,a,u,strength_wall,invDebyeWallGPU);
    u += x[ioffset+2] * weight;
    for(int j=0; j<n_blobs; j++){
      joffset = j * NDIM;
      // Compute vector between particles i and j    
      rx = x[ioffset    ] - x[joffset    ];
      ry = x[ioffset + 1] - x[joffset + 1];
      rz = x[ioffset + 2] - x[joffset + 2];
      if (Lx != 0){
        rx = rx - Lx*trunc(rx/Lx_over_2);
      }
      if (Ly != 0){
        ry = ry - Ly*trunc(ry/Ly_over_2);
      }
      //2. Compute particle-particle repulsion
      repulsionPart(rx, ry, rz, u, i,j, debye_part, kbT, epsilon, diam);
    }
    //LOOP END
    if(i == cross1_index){
      double r_eq = 1.0;
      double dx = x[ioffset] - x[cross2_index*3];
      double dy = x[ioffset+1] - x[cross2_index*3 + 1];
      double dz = x[ioffset+2] - x[cross2_index*3+ 2 ];
      cross_point_spring_potential(dx, dy, dz, u, kbT, r_eq);
    }
    else if(i == cross2_index){
      double r_eq = 1.0;
      double dx = x[ioffset] - x[cross1_index*3];
      double dy = x[ioffset+1] - x[cross1_index*3 + 1];
      double dz = x[ioffset+2] - x[cross1_index*3 + 2 ];
      cross_point_spring_potential(dx, dy, dz, u, kbT, r_eq);
    }
  }
  else
  {
    //make u large for blobs behind the wall
    u = 100000*(-(x[ioffset+2]-a) +1); //if a particle starts out of bounds somehow, then it won't want to move further out
  }
  //IF END
  //3. Save potential U_i
  total_U[i] = u;
  return;
}
""")


def many_body_potential(r_vectors, periodic_length, debye_wall, strength_wall, debye_part, weight, kbT, epsilon, cross1_index, cross2_index, diam, a):
   
    # Determine number of threads and blocks for the GPU
    utype = np.float64(1.)
    n_blobs = np.int32(len(r_vectors))
    threads_per_block=512
    if((n_blobs/threads_per_block) < 128):
        threads_per_block = 256
    if((n_blobs/threads_per_block) < 128):
        threads_per_block = 128
    if((n_blobs/threads_per_block) < 128):
        threads_per_block = 64
    if((n_blobs/threads_per_block) < 128):
        threads_per_block = 32

    num_blocks = (n_blobs-1)/threads_per_block + 1

    # Copy python info to simple numpy arrays
    x = np.zeros( n_blobs * 3)
    for i in range(n_blobs):       
        x[i*3    ] = r_vectors[i][0]
        x[i*3 + 1] = r_vectors[i][1]
        x[i*3 + 2] = r_vectors[i][2]
        
    # Allocate GPU memory
    x_gpu = cuda.mem_alloc(x.nbytes)
    u_gpu = cuda.mem_alloc(num_blocks * threads_per_block * utype.nbytes)
    
    # Copy data to the GPU (host to device)
    cuda.memcpy_htod(x_gpu, x)
    
    # Get pair interaction function
    pair_interactions = mod.get_function("potential_from_position")

    # Compute pair interactions
    pair_interactions(x_gpu, u_gpu,\
                      n_blobs,\
		      np.float64(periodic_length[0]),\
		      np.float64(periodic_length[1]),\
		      np.float64(debye_wall),\
		      np.float64(strength_wall),\
		      np.float64(debye_part),\
		      np.float64(weight),\
              np.float64(kbT),\
              np.float64(epsilon),\
              np.int32(cross1_index),\
              np.int32(cross2_index),\
              np.float64(diam), np.float64(a),\
		      block=(threads_per_block, 1, 1),\
		      grid=(num_blocks, 1)) 
    
    # Copy data from GPU to CPU (device to host)
    U = np.empty(n_blobs, dtype = float)
    cuda.memcpy_dtoh(U, u_gpu)
    #print U
    return np.sum(U)
