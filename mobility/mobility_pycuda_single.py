
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


mod = SourceModule("""
/*
 mobilityRPY computes the 3x3 RPY mobility
 between blobs i and j normalized with 8 pi eta a
*/
__device__ void mobilityRPY_single(float rx,
			    float ry,
			    float rz,
			    float &Mxx,
			    float &Mxy,
			    float &Mxz,
			    float &Myy,
			    float &Myz,
			    float &Mzz,
			    int i,
			    int j,
                            float invaGPU){
  
  float fourOverThree = 4.0 / 3.0;

  if(i==j){
    Mxx = fourOverThree;
    Mxy = 0;
    Mxz = 0;
    Myy = Mxx;
    Myz = 0;
    Mzz = Mxx;
  }
  else{
    rx = rx * invaGPU; //Normalize distance with hydrodynamic radius
    ry = ry * invaGPU;
    rz = rz * invaGPU;
    float r2 = rx*rx + ry*ry + rz*rz;
    float r = sqrt(r2);
    //We should not divide by zero but std::numeric_limits<float>::min() does not work in the GPU
    //float invr = (r > std::numeric_limits<float>::min()) ? (1.0 / r) : (1.0 / std::numeric_limits<float>::min())
    float invr = 1.0 / r;
    float invr2 = invr * invr;
    float c1, c2;
    if(r>=2){
      c1 = 1 + 2 / (3 * r2);
      c2 = (1 - 2 * invr2) * invr2;
      Mxx = (c1 + c2*rx*rx) * invr;
      Mxy = (     c2*rx*ry) * invr;
      Mxz = (     c2*rx*rz) * invr;
      Myy = (c1 + c2*ry*ry) * invr;
      Myz = (     c2*ry*rz) * invr;
      Mzz = (c1 + c2*rz*rz) * invr;
    }
    else{
      c1 = fourOverThree * (1 - 0.28125 * r); // 9/32 = 0.28125
      c2 = fourOverThree * 0.09375 * invr;    // 3/32 = 0.09375
      Mxx = c1 + c2 * rx*rx ;
      Mxy =      c2 * rx*ry ;
      Mxz =      c2 * rx*rz ;
      Myy = c1 + c2 * ry*ry ;
      Myz =      c2 * ry*rz ;
      Mzz = c1 + c2 * rz*rz ;
    }
  } 
  
  return;
}


/*
 mobilityRPY computes the 3x3 mobility correction due to a wall
 between blobs i and j normalized with 8 pi eta a.
 This uses the expression from the Swan and Brady paper for a finite size particle.
 Mobility is normalize by 8*pi*eta*a.
*/
__device__ void mobilitySingleWallCorrection_single(float rx,
			                     float ry,
			                     float rz,
			                     float &Mxx,
                  			     float &Mxy,
			                     float &Mxz,
                                             float &Myx,
			                     float &Myy,
			                     float &Myz,
                                             float &Mzx,
                                             float &Mzy,
			                     float &Mzz,
			                     int i,
			                     int j,
                                             float invaGPU,
                                             float hj){
  if(i == j){
    float invZi = 1.0 / hj;
    Mxx += -(9*invZi - 2*pow(invZi,3) + pow(invZi,5)) / 12.0;
    Myy += -(9*invZi - 2*pow(invZi,3) + pow(invZi,5)) / 12.0;
    Mzz += -(9*invZi - 4*pow(invZi,3) + pow(invZi,5)) / 6.0;
  }
  else{
    float h_hat = hj / rz;
    float invR = rsqrt(rx*rx + ry*ry + rz*rz); // = 1 / r;
    float ex = rx * invR;
    float ey = ry * invR;
    float ez = rz * invR;
    
    float fact1 = -(3*(1+2*h_hat*(1-h_hat)*ez*ez) * invR + 2*(1-3*ez*ez) * pow(invR,3) - 2*(1-5*ez*ez) * pow(invR,5))  / 3.0;
    float fact2 = -(3*(1-6*h_hat*(1-h_hat)*ez*ez) * invR - 6*(1-5*ez*ez) * pow(invR,3) + 10*(1-7*ez*ez) * pow(invR,5)) / 3.0;
    float fact3 =  ez * (3*h_hat*(1-6*(1-h_hat)*ez*ez) * invR - 6*(1-5*ez*ez) * pow(invR,3) + 10*(2-7*ez*ez) * pow(invR,5)) * 2.0 / 3.0;
    float fact4 =  ez * (3*h_hat*invR - 10*pow(invR,5)) * 2.0 / 3.0;
    float fact5 = -(3*h_hat*h_hat*ez*ez*invR + 3*ez*ez*pow(invR, 3) + (2-15*ez*ez)*pow(invR, 5)) * 4.0 / 3.0;
    
    Mxx += fact1 + fact2 * ex*ex;
    Mxy += fact2 * ex*ey;
    Mxz += fact2 * ex*ez + fact3 * ex;
    Myx += fact2 * ey*ex;
    Myy += fact1 + fact2 * ey*ey;
    Myz += fact2 * ey*ez + fact3 * ey;
    Mzx += fact2 * ez*ex + fact4 * ex;
    Mzy += fact2 * ez*ey + fact4 * ey;
    Mzz += fact1 + fact2 * ez*ez + fact3 * ez + fact4 * ez + fact5;         
  }

}




/*
 velocity_from_force computes the product
 U = M*F
*/
__global__ void velocity_from_force_single(const float *x,
                                    const float *f,					
                                    float *u,
				    int number_of_blobs,
                                    float eta,
                                    float a){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;   

  float invaGPU = 1.0 / a;

  float Ux=0;
  float Uy=0;
  float Uz=0;

  float rx, ry, rz;

  float Mxx, Mxy, Mxz;
  float Myx, Myy, Myz;
  float Mzx, Mzy, Mzz;

  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  for(int j=0; j<number_of_blobs; j++){
    joffset = j * NDIM;

    // Compute vector between particles i and j
    rx = x[ioffset    ] - x[joffset    ];
    ry = x[ioffset + 1] - x[joffset + 1];
    rz = x[ioffset + 2] - x[joffset + 2];

    // 1. Compute mobility for pair i-j
    mobilityRPY_single(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, invaGPU);
    Myx = Mxy;
    Mzx = Mxz;
    Mzy = Myz;
    mobilitySingleWallCorrection_single(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy,Mzz, i,j, invaGPU, x[joffset+2]/a);

    //2. Compute product M_ij * F_j
    Ux = Ux + (Mxx * f[joffset] + Mxy * f[joffset + 1] + Mxz * f[joffset + 2]);
    Uy = Uy + (Myx * f[joffset] + Myy * f[joffset + 1] + Myz * f[joffset + 2]);
    Uz = Uz + (Mzx * f[joffset] + Mzy * f[joffset + 1] + Mzz * f[joffset + 2]);
  }
  //LOOP END

  //3. Save velocity U_i
  float pi = 4.0 * atan(1.0);
  u[ioffset    ] = Ux / (8 * pi * eta * a);
  u[ioffset + 1] = Uy / (8 * pi * eta * a);
  u[ioffset + 2] = Uz / (8 * pi * eta * a);

  return;
}


__global__ void velocity_from_force_RPY_2_single(const float *x,
                                          const float *f,
                                          float *u,
                                          int number_of_blobs,
                                          float eta,
                                          float a){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  return;

}
""")

def single_wall_mobility_times_force_pycuda_single(r_vectors, force, eta, a):
   
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
    x = np.zeros( number_of_blobs * 3, dtype=np.float32)
    f = np.zeros( number_of_blobs * 3, dtype=np.float32)
    for i in range(number_of_blobs):
        #x[i*3    ] = r_vectors[i][0]
        #x[i*3 + 1] = r_vectors[i][1]
        #x[i*3 + 2] = r_vectors[i][2]
        #f[i*3]     = force[i,0]
        #f[i*3 + 1] = force[i,1]
        #f[i*3 + 2] = force[i,2]
        
        x[i*3    ] = r_vectors[i][0]
        x[i*3 + 1] = r_vectors[i][1]
        x[i*3 + 2] = r_vectors[i][2]
        f[i*3]     = force[i*3]
        f[i*3 + 1] = force[i*3+1]
        f[i*3 + 2] = force[i*3+2]
        
        
    # Allocate GPU memory
    x_gpu = cuda.mem_alloc(x.nbytes)
    f_gpu = cuda.mem_alloc(f.nbytes)
    u_gpu = cuda.mem_alloc(f.nbytes)
    number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)
    
    # Copy data to the GPU (host to device)
    cuda.memcpy_htod(x_gpu, x)
    cuda.memcpy_htod(f_gpu, f)
    
    # Get mobility function
    mobility = mod.get_function("velocity_from_force_single")

    # Compute mobility force product
    mobility(x_gpu, f_gpu, u_gpu, number_of_blobs, np.float32(eta), np.float32(a), block=(threads_per_block, 1, 1), grid=(num_blocks, 1)) 
    
    # Copy data from GPU to CPU (device to host)
    u = np.empty_like(f)
    cuda.memcpy_dtoh(u, u_gpu)


    return u


