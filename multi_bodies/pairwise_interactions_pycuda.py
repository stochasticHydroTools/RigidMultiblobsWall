
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


mod = SourceModule("""
 
////////// Wall repulsion //////////////////////////////////////////////////
__device__ void repulsionWall(double h,
			      double &fz,
			      double strength_wall,
                              double invDebyeWallGPU){
   
  double h2 = h*h;
  double invh2 = 1.0 / h2;
  fz = strength_wall*(h*invDebyeWallGPU+1.0)*exp(-h*invDebyeWallGPU)*invh2;
 
  return;
}

////////// Part-part repulsion //////////////////////////////////////////////////

__device__ void repulsionPart(double rx,
			      double ry,
			      double rz,
			      double &fx,
			      double &fy,
			      double &fz,
			      int i,
			      int j,
			      double invDebyePartGPU,
			      double strength_part,
			      double lower_bound,
			      double upper_bound,
			      double diam){
				
  
  if(i != j){
    double r = sqrt(rx*rx + ry*ry + rz*rz);
    double invR = 1.0 / r;
    double invR2 = invR * invR;
    double r_diam = r - diam;
   
    
    if (r_diam<upper_bound && r_diam>lower_bound){
      double fact = strength_part*diam*invR2*(invDebyePartGPU + invR)*exp(-r_diam*invDebyePartGPU);
      fx += fact*rx;
      fy += fact*ry;
      fz += fact*rz;
    }
  }

}


/*
 force_from_position computes paiwise and wall interactions
*/
__global__ void force_from_position(const double *x,
                                    double *f,					
				    int number_of_blobs,
				    double debye_wall,
				    double strength_wall,
				    double debye_part,
				    double strength_part,
				    double gravity,
                                    double diam,
                                    double a){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;   

  double invDebyeWallGPU = 1.0 / debye_wall;
  double invDebyePartGPU = 1.0 / debye_part;
  double upper_bound = 1.0*diam;
  double lower_bound = -0.2*diam;
  double fx=0;
  double fy=0;
  double fz=0;

  double rx, ry, rz;

  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  if (x[ioffset+2]>a){
    // 1. Compute repulsion from the wall
    repulsionWall(x[ioffset+2]-a,fz,strength_wall,invDebyeWallGPU);
  
    for(int j=0; j<number_of_blobs; j++){
      joffset = j * NDIM;

      // Compute vector between particles i and j
      rx = x[ioffset    ] - x[joffset    ];
      ry = x[ioffset + 1] - x[joffset + 1];
      rz = x[ioffset + 2] - x[joffset + 2];
      
      //2. Compute particle-particle repulsion
      repulsionPart(rx, ry, rz, fx,fy,fz, i,j, invDebyePartGPU, strength_part, lower_bound, upper_bound, diam);
    }
    //LOOP END
  }
  //IF END
  //3. Save force F_i
  f[ioffset    ] = fx;
  f[ioffset + 1] = fy;
  f[ioffset + 2] = fz + gravity;

  return;
}

""")

def particle_pair_interaction_pycuda(r_vectors,\
                                     debye_wall,\
				     strength_wall,\
				     debye_part,\
				     strength_part,\
				     gravity,\
				     diam, a):
   
    # Determine number of threads and blocks for the GPU
    number_of_blobs = np.int32(len(r_vectors))
    threads_per_block=512
    if((number_of_blobs/threads_per_block) < 128):
        threads_per_block = 256
    if((number_of_blobs/threads_per_block) < 128):
        threads_per_block = 128
    if((number_of_blobs/threads_per_block) < 128):
        threads_per_block = 64
    if((number_of_blobs/threads_per_block) < 128):
        threads_per_block = 32
    num_blocks = (number_of_blobs-1)/threads_per_block + 1

    # Copy python info to simple numpy arrays
    x = np.zeros( number_of_blobs * 3)
    for i in range(number_of_blobs):       
        x[i*3    ] = r_vectors[i][0]
        x[i*3 + 1] = r_vectors[i][1]
        x[i*3 + 2] = r_vectors[i][2]
        
        
    # Allocate GPU memory
    x_gpu = cuda.mem_alloc(x.nbytes)
    f_gpu = cuda.mem_alloc(x.nbytes)
    number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)
    
    # Copy data to the GPU (host to device)
    cuda.memcpy_htod(x_gpu, x)
    
    # Get pair interaction function
    pair_interactions = mod.get_function("force_from_position")

    # Compute pair interactions
    pair_interactions(x_gpu, f_gpu,\
                      number_of_blobs,\
		      np.float64(debye_wall),\
		      np.float64(strength_wall),\
		      np.float64(debye_part),\
		      np.float64(strength_part),\
		      np.float64(gravity),\
	              np.float64(diam), np.float64(a),\
		      block=(threads_per_block, 1, 1),\
		      grid=(num_blocks, 1)) 
    
    # Copy data from GPU to CPU (device to host)
    f = np.empty_like(x)
    cuda.memcpy_dtoh(f, f_gpu)


    return f


