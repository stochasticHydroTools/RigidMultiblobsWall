#include "icc_cuda.h"

#define chkErrq(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    // printf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    cout << "GPUasser: " << cudaGetErrorString(code) << "   "  << file << "  "  << line << endl;
    if (abort) exit(code);
  }
}


/*
  Constructor: build the sparse mobility matrix M
  and compute the Cholesky factorization M=L*L.T
  where L is a lower triangular matrix.
*/
icc::icc(const double blob_radius, 
	 const double eta, 
	 const int number_of_blobs,
	 const double *x){
  d_blob_radius = blob_radius;
  d_eta = eta;
  d_number_of_blobs = number_of_blobs;
  d_x = x;

  // Determine number of blocks and threads for the GPU
  d_threads_per_block = 512;
  if((d_number_of_blobs / d_threads_per_block) < 512){
    d_threads_per_block = 256;
  }
  if((d_number_of_blobs / d_threads_per_block) < 256){
    d_threads_per_block = 128;
  }
  if((d_number_of_blobs / d_threads_per_block) < 128){
    d_threads_per_block = 128;
  }
  if((d_number_of_blobs / d_threads_per_block) < 128){
    d_threads_per_block = 64;
  }
  if((d_number_of_blobs / d_threads_per_block) < 32){
    d_threads_per_block = 128;
  }
  d_num_blocks = (d_number_of_blobs - 1) / d_threads_per_block + 1;
}

/*
  Destructor: free memory on the GPU and CPU.
*/
icc::~icc(){
  // Free GPU memory
  chkErrq(cudaFree(d_x_gpu));
}

/*
  Build sparse mobility matrix M.
*/
int icc::buildSparseMobilityMatrix(){
  // Allocate GPU memory
  chkErrq(cudaMalloc((void**)&d_x_gpu, d_number_of_blobs * 3 *sizeof(double)));

  // Build sparse mobility matrix
  
  return 0;
}

/*
 mobilityUFRPY computes the 3x3 RPY mobility
 between blobs i and j normalized with 8 pi eta a
*/
__device__ void mobilityUFRPY(double rx,
			      double ry,
			      double rz,
			      double &Mxx,
			      double &Mxy,
			      double &Mxz,
			      double &Myy,
			      double &Myz,
			      double &Mzz,
			      int i,
			      int j,
                              double invaGPU){
  
  double fourOverThree = 4.0 / 3.0;

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
    double r2 = rx*rx + ry*ry + rz*rz;
    double r = sqrt(r2);
    //We should not divide by zero but std::numeric_limits<double>::min() does not work in the GPU
    //double invr = (r > std::numeric_limits<double>::min()) ? (1.0 / r) : (1.0 / std::numeric_limits<double>::min())
    double invr = 1.0 / r;
    double invr2 = invr * invr;
    double c1, c2;
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
__device__ void mobilityUFSingleWallCorrection(double rx,
			                       double ry,
			                       double rz,
			                       double &Mxx,
                  			       double &Mxy,
			                       double &Mxz,
                                               double &Myx,
			                       double &Myy,
			                       double &Myz,
                                               double &Mzx,
                                               double &Mzy,
			                       double &Mzz,
			                       int i,
			                       int j,
                                               double invaGPU,
                                               double hj){
  if(i == j){
    double invZi = 1.0 / hj;
    Mxx += -(9*invZi - 2*pow(invZi,3) + pow(invZi,5)) / 12.0;
    Myy += -(9*invZi - 2*pow(invZi,3) + pow(invZi,5)) / 12.0;
    Mzz += -(9*invZi - 4*pow(invZi,3) + pow(invZi,5)) / 6.0;
  }
  else{
    double h_hat = hj / rz;
    double invR = rsqrt(rx*rx + ry*ry + rz*rz); // = 1 / r;
    double ex = rx * invR;
    double ey = ry * invR;
    double ez = rz * invR;
    
    double fact1 = -(3*(1+2*h_hat*(1-h_hat)*ez*ez) * invR + 2*(1-3*ez*ez) * pow(invR,3) - 2*(1-5*ez*ez) * pow(invR,5))  / 3.0;
    double fact2 = -(3*(1-6*h_hat*(1-h_hat)*ez*ez) * invR - 6*(1-5*ez*ez) * pow(invR,3) + 10*(1-7*ez*ez) * pow(invR,5)) / 3.0;
    double fact3 =  ez * (3*h_hat*(1-6*(1-h_hat)*ez*ez) * invR - 6*(1-5*ez*ez) * pow(invR,3) + 10*(2-7*ez*ez) * pow(invR,5)) * 2.0 / 3.0;
    double fact4 =  ez * (3*h_hat*invR - 10*pow(invR,5)) * 2.0 / 3.0;
    double fact5 = -(3*h_hat*h_hat*ez*ez*invR + 3*ez*ez*pow(invR, 3) + (2-15*ez*ez)*pow(invR, 5)) * 4.0 / 3.0;
    
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
  Determine number of non-zero elements (nnz)
*/
__global__ void countNnz(const double *x, unsigned long long int *nnzGPU, const double cutoff, const int N){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= N) return;   

  double rx, ry, rz, r2;
  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  // Loop over columns
  for(int j=0; j<N; j++){
    joffset = j * NDIM;
    
    // Compute vector between blobs i and j
    rx = x[ioffset    ] - x[joffset    ];
    ry = x[ioffset + 1] - x[joffset + 1];
    rz = x[ioffset + 2] - x[joffset + 2];
    r2 = (rx*rx + ry*ry + rz*rz);
    
    // If blobs are close increse nnz
    if(r2 < cutoff*cutoff){
      atomicAdd(nnzGPU, 9);
    }
  }
}

/*
  Build a sparse matrix with coordinated format (COO). See cuSparse documentation.
*/
__global__ void buildCOOMatrix(const double *x,
			       double *cooValA,
                               int *cooRowIndA,
                               int *cooColIndA,
                               unsigned long long int *nnzGPU,
			       const double eta,
			       const double a,
			       const double cutoff,
			       const int N){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= N) return;   

  double pi = 4.0 * atan(1.0);
  double norm_fact = 1.0 / (8 * pi * eta * a);  
  double inva = 1.0 / a;
  double rx, ry, rz, r2;
  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  double Mxx, Mxy, Mxz;
  double Myx, Myy, Myz;
  double Mzx, Mzy, Mzz;

  // Loop over columns
  for(int j=0; j<N; j++){
    joffset = j * NDIM;
    
    // Compute vector between blobs i and j
    rx = x[ioffset    ] - x[joffset    ];
    ry = x[ioffset + 1] - x[joffset + 1];
    rz = x[ioffset + 2] - x[joffset + 2];
    r2 = (rx*rx + ry*ry + rz*rz);
    
    // If blobs are close compute pair-mobility
    if(r2 < cutoff*cutoff){
      mobilityUFRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, inva);
      Myx = Mxy;
      Mzx = Mxz;
      Mzy = Myz;
      mobilityUFSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy,Mzz, i,j, inva, x[joffset+2]/a);
      
      int nnz_old = atomicAdd(nnzGPU, 9);      
      cooValA[nnz_old] = Mxx * norm_fact;
      cooRowIndA[nnz_old] = ioffset;
      cooColIndA[nnz_old] = joffset;

      nnz_old++;
      cooValA[nnz_old] = Mxy * norm_fact;
      cooRowIndA[nnz_old] = ioffset;
      cooColIndA[nnz_old] = joffset + 1;

      nnz_old++;
      cooValA[nnz_old] = Mxz * norm_fact;
      cooRowIndA[nnz_old] = ioffset;
      cooColIndA[nnz_old] = joffset + 2;

      nnz_old++;
      cooValA[nnz_old] = Myx * norm_fact;
      cooRowIndA[nnz_old] = ioffset + 1;
      cooColIndA[nnz_old] = joffset;

      nnz_old++;
      cooValA[nnz_old] = Myy * norm_fact;
      cooRowIndA[nnz_old] = ioffset + 1;
      cooColIndA[nnz_old] = joffset + 1;

      nnz_old++;
      cooValA[nnz_old] = Myz * norm_fact;
      cooRowIndA[nnz_old] = ioffset + 1;
      cooColIndA[nnz_old] = joffset + 2;

      nnz_old++;
      cooValA[nnz_old] = Mzx * norm_fact;
      cooRowIndA[nnz_old] = ioffset + 2;
      cooColIndA[nnz_old] = joffset ;

      nnz_old++;
      cooValA[nnz_old] = Mzy * norm_fact;
      cooRowIndA[nnz_old] = ioffset + 2;
      cooColIndA[nnz_old] = joffset + 1;

      nnz_old++;
      cooValA[nnz_old] = Mzz * norm_fact;
      cooRowIndA[nnz_old] = ioffset + 2;
      cooColIndA[nnz_old] = joffset + 2;
    } 
  }  
} 


int main(){

  // Define parameters
  int status;
  double blob_radius = 1.0;
  double eta = 1.0;
  int number_of_blobs = 2;

  // Create CPU arrays
  double *x = new double [number_of_blobs * 3];
  for(int i=0; i<(number_of_blobs * 3); i++){
    x[i] = 10.0 * rand() / RAND_MAX;
    cout << i << "  " << x[i] << endl;
  }
 
  // Create icc object
  icc icc_obj = icc(blob_radius, eta, number_of_blobs, x);
  
  // Build sparse mobility matrix
  status = icc_obj.buildSparseMobilityMatrix();
  cout << "Build sparse mobility matrix = " << status << endl;
  

  // Free CPU memory
  delete[] x;
  cout << "# End" << endl;
  return 0;
}
