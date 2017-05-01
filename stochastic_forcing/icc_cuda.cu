#include "icc_cuda.h"
#include <fstream>
#include <thrust/version.h>
// #include <thrust/reduce.h>
// #include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
// #include <thrust/transform_reduce.h>
// #include <thrust/functional.h>
#include <thrust/sort.h>
// #include </usr/include/python2.6/Python.h>

#define chkErrq(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    cout << "GPUasser: " << cudaGetErrorString(code) << "   "  << file << "  "  << line << endl;
    if (abort) exit(code);
  }
}

#define chkErrqCusparse(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(cusparseStatus_t code, const char *file, int line, bool abort=true)
{
  if (code != 0) 
  {
    if(code == 1)
      cout << code << " cusparseStatusNotInitialized " << file << "  " << line << endl;
    else if(code == 2)
      cout << code << " cusparseStatusAllocFailed " << file << "  " << line << endl;
    else if (code == 3)
      cout << code << " cusparseStatusInvalidValue " << file << "  " << line << endl;
    else if (code == 4)
      cout << code << " cusparseStatusArchMismatch " << file << "  " << line << endl;
    else if (code == 5)
      cout << code << " cusparseStatusMappingError " << file << "  " << line << endl;
    else if (code == 6)
      cout << code << " cusparseStatusExecutionFailed " << file << "  " << line << endl;
    else if (code == 7)
      cout << code << " cusparseStatusInternalError " << file << "  " << line << endl;
    else if (code == 8)
      cout << code << " cusparseStatusMatrixTypeNotSupported " << file << "  " << line << endl;
    // cout << "cuSparseasser: " << code << "   "  << file << "  "  << line << endl;
    if (abort) exit(code);
  }
}

struct saxpy_functor
{
  const int m;
  saxpy_functor(int _m) : m(_m) {}

    __host__ __device__
    unsigned long long int operator()(const int& x, const unsigned long long int& y) const { 
      return (unsigned long long) m * (unsigned long long) x + y;
    }
};

void saxpy_fast(int m, thrust::device_vector<int>& X, thrust::device_vector<unsigned long long int>& Y)
{
  // Y <- m * X + Y
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(m));
}

int print_csr_matrix_in_dense_format(cusparseHandle_t handle, 
				     int num_rows, 
				     int num_col, 
				     int nnz,
				     const cusparseMatDescr_t descr,
				     const double *csrVal, 
				     const int *csrRowPtr,
				     const int *csrColInd,
				     const double *x){
  int size = num_rows * num_col;
  // Allocate memory
  double *A = new double [size];
  double *A_gpu;
  chkErrq(cudaMalloc((void**)&A_gpu, size * sizeof(double)));
  // Copy matrix to dense format and print
  cusparseMatrixType_t mat_type = cusparseGetMatType(descr);
  chkErrqCusparse(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  cusparseDcsr2dense(handle, num_rows, num_col, descr, csrVal, csrRowPtr, csrColInd, A_gpu, num_rows);
  chkErrqCusparse(cusparseSetMatType(descr, mat_type));
  chkErrq(cudaMemcpy(A, A_gpu, size * sizeof(double), cudaMemcpyDeviceToHost));  
  printf("Matrix = \n");
  for(int i=0; i<num_rows; i++){
    for(int j=0; j<num_col; j++){
      printf("%010f  ", A[i*num_col + j]);
    }
    printf("\n");
  }
  printf("\n\n");
  // Free memory
  chkErrq(cudaFree(A_gpu));
  delete[] A;  
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
 velocity_from_force computes the product
 U = M*F
*/
__global__ void velocity_from_force(const double *x,
                                    const double *f,					
                                    double *u,
				    int number_of_blobs,
                                    double eta,
                                    double a){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;   

  double invaGPU = 1.0 / a;

  double Ux=0;
  double Uy=0;
  double Uz=0;

  double rx, ry, rz;

  double Mxx, Mxy, Mxz;
  double Myx, Myy, Myz;
  double Mzx, Mzy, Mzz;

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
    mobilityUFRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, invaGPU);
    Myx = Mxy;
    Mzx = Mxz;
    Mzy = Myz;
    mobilityUFSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy,Mzz, i,j, invaGPU, x[joffset+2]/a);

    //2. Compute product M_ij * F_j
    Ux = Ux + (Mxx * f[joffset] + Mxy * f[joffset + 1] + Mxz * f[joffset + 2]);
    Uy = Uy + (Myx * f[joffset] + Myy * f[joffset + 1] + Myz * f[joffset + 2]);
    Uz = Uz + (Mzx * f[joffset] + Mzy * f[joffset + 1] + Mzz * f[joffset + 2]);
  }
  //LOOP END

  //3. Save velocity U_i
  double pi = 4.0 * atan(1.0);
  double norm_fact_f = 8 * pi * eta * a;
  u[ioffset    ] = Ux / norm_fact_f;
  u[ioffset + 1] = Uy / norm_fact_f;
  u[ioffset + 2] = Uz / norm_fact_f;

  return;
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
      unsigned long long int nnz_old = atomicAdd(nnzGPU, 9);
    }
  }
}


/*
  Build a sparse lower triangular matrix with coordinated format (COO). See cuSparse documentation.
*/
__global__ void buildLowerTriangularCOOMatrix(const double *x,
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
  for(int j=i; j<N; j++){
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
      
      if( i != j){
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
      else{
        int nnz_old = atomicAdd(nnzGPU, 6); //
        cooValA[nnz_old] = Mxx * norm_fact;
        cooRowIndA[nnz_old] = ioffset;
        cooColIndA[nnz_old] = joffset;

	nnz_old++; //
	cooValA[nnz_old] = Mxy * norm_fact; //
	cooRowIndA[nnz_old] = ioffset;      //
	cooColIndA[nnz_old] = joffset + 1;  //

	nnz_old++; //
	cooValA[nnz_old] = Mxz * norm_fact; //
	cooRowIndA[nnz_old] = ioffset; //
	cooColIndA[nnz_old] = joffset + 2; //

        nnz_old++;
        cooValA[nnz_old] = Myy * norm_fact;
        cooRowIndA[nnz_old] = ioffset + 1;
        cooColIndA[nnz_old] = joffset + 1;

	nnz_old++; //
	cooValA[nnz_old] = Myz * norm_fact; //
	cooRowIndA[nnz_old] = ioffset + 1; //
	cooColIndA[nnz_old] = joffset + 2; //

        nnz_old++;
        cooValA[nnz_old] = Mzz * norm_fact;
        cooRowIndA[nnz_old] = ioffset + 2;
        cooColIndA[nnz_old] = joffset + 2;
      }
    } 
  }  
} 


/*
  Determine number of non-zero elements (nnz) in a lower
  Triangular matrix.
*/
__global__ void countLowerTriangularNnz(const double *x, unsigned long long int *nnzGPU, const double cutoff, const int N){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= N) return;   

  double rx, ry, rz, r2;
  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  // Loop over columns
  for(int j=i; j<N; j++){
    joffset = j * NDIM;
    
    // Compute vector between blobs i and j
    rx = x[ioffset    ] - x[joffset    ];
    ry = x[ioffset + 1] - x[joffset + 1];
    rz = x[ioffset + 2] - x[joffset + 2];
    r2 = (rx*rx + ry*ry + rz*rz);
    
    // If blobs are close increse nnz
    if(r2 < cutoff*cutoff){
      if( i != j){
        atomicAdd(nnzGPU, 9);
      }
      else{
        atomicAdd(nnzGPU, 6); //
      }
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


/*
  Constructor: build the sparse mobility matrix M
  and compute the Cholesky factorization M=L*L.T
  where L is a lower triangular matrix.
*/
icc::icc(const double blob_radius, 
	 const double eta, 
	 const double cutoff,
	 const int number_of_blobs,
	 const double *x){
  d_icc_is_initialized = 0;
  d_blob_radius = blob_radius;
  d_eta = eta;
  d_cutoff = cutoff;
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
  Constructor: build the sparse mobility matrix M
  and compute the Cholesky factorization M=L*L.T
  where L is a lower triangular matrix.
*/
icc::icc(const double blob_radius, 
	 const double eta, 
	 const double cutoff,
	 const int number_of_blobs,
	 bp::object x_obj){
  d_icc_is_initialized = 0;
  d_blob_radius = blob_radius;
  d_eta = eta;
  d_cutoff = cutoff;
  d_number_of_blobs = number_of_blobs;

  // Extract pointer
  PyObject* pobj = x_obj.ptr();
  Py_buffer pybuf;
  PyObject_GetBuffer(pobj, &pybuf, PyBUF_SIMPLE);
  void *buf = pybuf.buf;
  double *x = (double*)buf;
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
  // Delete cusparse objects
  cout << "~icc STARTS " << endl;
  if(d_icc_is_initialized){
    chkErrqCusparse(cusparseDestroySolveAnalysisInfo(d_info_LT)); 
    chkErrqCusparse(cusparseDestroySolveAnalysisInfo(d_info_L)); 
    chkErrqCusparse(cusparseDestroySolveAnalysisInfo(d_info_M)); 
    cusparseDestroyMatDescr(d_descr_L);
    cusparseDestroyMatDescr(d_descr_M);
    chkErrqCusparse(cusparseDestroy(d_cusp_handle));

    // Free GPU memory
    chkErrq(cudaFree(d_x_gpu));
    chkErrq(cudaFree(d_nnz_gpu));
    chkErrq(cudaFree(d_aux_gpu));
    chkErrq(cudaFree(d_cooVal_gpu));
    chkErrq(cudaFree(d_cooVal_sorted_gpu));
    chkErrq(cudaFree(d_cooRowInd_gpu));
    chkErrq(cudaFree(d_cooColInd_gpu));
    chkErrq(cudaFree(d_csrRowPtr_gpu));
  }
  cout << "~icc DONE" << endl;
}

/*
  Build sparse mobility matrix M.
*/
int icc::init_icc(){
  int N = d_number_of_blobs * 3;

  // Allocate GPU memory
  chkErrq(cudaMalloc((void**)&d_x_gpu, N * sizeof(double)));
  chkErrq(cudaMalloc((void**)&d_nnz_gpu, sizeof(unsigned long long int)));
  chkErrq(cudaMalloc((void**)&d_aux_gpu, N * sizeof(double))); 
 
  // Copy data from CPU to GPU
  chkErrq(cudaMemcpy(d_x_gpu, d_x, N * sizeof(double), cudaMemcpyHostToDevice));
  d_nnz = 0;
  chkErrq(cudaMemcpy(d_nnz_gpu, &d_nnz, sizeof(unsigned long long int), cudaMemcpyHostToDevice));

  // Count non-zero elements in mobility matrix
  countLowerTriangularNnz<<<d_num_blocks, d_threads_per_block>>>(d_x_gpu, d_nnz_gpu, d_cutoff, d_number_of_blobs);
  // countNnz<<<d_num_blocks, d_threads_per_block>>>(d_x_gpu, d_nnz_gpu, d_cutoff, d_number_of_blobs);
  chkErrq(cudaPeekAtLastError());
  chkErrq(cudaMemcpy(&d_nnz, d_nnz_gpu, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
  cout << "nnz = " << d_nnz << endl;

  // Allocate GPU memory for the sparse mobility matrix
  chkErrq(cudaMalloc((void**)&d_cooVal_gpu, d_nnz * sizeof(double)));
  chkErrq(cudaMalloc((void**)&d_cooVal_sorted_gpu, d_nnz * sizeof(double)));
  chkErrq(cudaMalloc((void**)&d_cooRowInd_gpu, d_nnz * sizeof(int)));
  chkErrq(cudaMalloc((void**)&d_cooColInd_gpu, d_nnz * sizeof(int)));
  chkErrq(cudaMalloc((void**)&d_csrRowPtr_gpu, ((3 * d_number_of_blobs) + 1) * sizeof(int)));

  // Build sparse mobility matrix
  d_nnz = 0;
  chkErrq(cudaMemcpy(d_nnz_gpu, &d_nnz, sizeof(unsigned long long int), cudaMemcpyHostToDevice));
  if(1){
    buildLowerTriangularCOOMatrix<<<d_num_blocks, d_threads_per_block>>>(d_x_gpu,
									 d_cooVal_gpu,
									 d_cooRowInd_gpu,
									 d_cooColInd_gpu,
									 d_nnz_gpu,
									 d_eta,
									 d_blob_radius,
									 d_cutoff,
									 d_number_of_blobs);
  }
  else{
    buildCOOMatrix<<<d_num_blocks, d_threads_per_block>>>(d_x_gpu,
							  d_cooVal_gpu,
							  d_cooRowInd_gpu,
							  d_cooColInd_gpu,
							  d_nnz_gpu,
							  d_eta,
							  d_blob_radius,
							  d_cutoff,
							  d_number_of_blobs);
  }
  chkErrq(cudaPeekAtLastError());
  chkErrq(cudaMemcpy(&d_nnz, d_nnz_gpu, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
  cout << "nnz = " << d_nnz << endl;
  
  // Init cuSparse
  chkErrqCusparse(cusparseCreate(&d_cusp_handle));
  d_base = cusparseIndexBase_t(0);
  
  // Sort matrix to COO format
  {
    thrust::device_vector<int> vec_col(d_cooColInd_gpu, d_cooColInd_gpu + d_nnz);
    chkErrq(cudaPeekAtLastError());
    thrust::device_vector<int> vec_row(d_cooRowInd_gpu, d_cooRowInd_gpu + d_nnz);
    chkErrq(cudaPeekAtLastError());
    thrust::device_vector<double> vec_val(d_cooVal_gpu, d_cooVal_gpu + d_nnz);
    chkErrq(cudaPeekAtLastError());
    thrust::device_vector<int> vec_col_sorted(d_nnz);
    chkErrq(cudaPeekAtLastError());
    thrust::device_vector<int> vec_row_sorted(d_nnz);
    chkErrq(cudaPeekAtLastError());
    thrust::device_vector<double> vec_val_sorted(d_nnz);
    chkErrq(cudaPeekAtLastError());
    thrust::device_vector<unsigned long long int> vec_global_index(d_cooColInd_gpu, d_cooColInd_gpu + d_nnz);
    chkErrq(cudaPeekAtLastError());
    if(0){
      cout << "Print values  ";
      thrust::copy(vec_val.begin(), vec_val.end(), std::ostream_iterator<double>(std::cout, " "));
      cout << endl;
      cout << "Print columns ";
      thrust::copy(vec_global_index.begin(), vec_global_index.end(), std::ostream_iterator<unsigned long long int>(std::cout, " "));
      cout << endl;
      cout << "Print rows    ";
      thrust::copy(vec_row.begin(), vec_row.end(), std::ostream_iterator<int>(std::cout, " "));
      cout << endl;
      // thrust::sort(d_cooRowInd, d_cooRowInd + d_nnz);
    }
    // Create global index = row*N + col
    saxpy_fast(N, vec_row, vec_global_index);
    if(0){
      // thrust::host_vector<unsigned long long int> vec_global_index_host = vec_global_index;
      cout << "Print index  ";
      thrust::copy(vec_global_index.begin(), vec_global_index.end(), std::ostream_iterator<unsigned long long int>(std::cout, " "));
      cout << endl;
    }

    // Initialize vector to [0, 1, 2, ...]
    thrust::counting_iterator<int> iter(0);
    chkErrq(cudaPeekAtLastError());
    thrust::device_vector<int> indices(d_nnz);
    chkErrq(cudaPeekAtLastError());
    thrust::copy(iter, iter + indices.size(), indices.begin());
    chkErrq(cudaPeekAtLastError());
    // Sort the indices using the global index as the key
    thrust::sort_by_key(vec_global_index.begin(), vec_global_index.end(), indices.begin());
    chkErrq(cudaPeekAtLastError());
    if(0){
      cout << "Print index  ";
      thrust::copy(vec_global_index.begin(), vec_global_index.end(), std::ostream_iterator<unsigned long long int>(std::cout, "  "));
      cout << endl;
    }

    // Sort rows, columns and values with the indices
    thrust::gather(indices.begin(), indices.end(), vec_col.begin(), vec_col_sorted.begin());
    thrust::gather(indices.begin(), indices.end(), vec_row.begin(), vec_row_sorted.begin());
    thrust::gather(indices.begin(), indices.end(), vec_val.begin(), vec_val_sorted.begin());
    if(0){
      cout << endl << endl << endl;
      cout << "Print columns ";
      thrust::copy(vec_col_sorted.begin(), vec_col_sorted.end(), std::ostream_iterator<int>(std::cout, " "));
      cout << endl;
      cout << "Print rows    ";
      thrust::copy(vec_row_sorted.begin(), vec_row_sorted.end(), std::ostream_iterator<int>(std::cout, " "));
      cout << endl;
      cout << "Print values  ";
      thrust::copy(vec_val_sorted.begin(), vec_val_sorted.end(), std::ostream_iterator<double>(std::cout, " "));
      cout << endl;
    }
    
    // Copy thrust vectors to arrays
    thrust::copy(vec_col_sorted.begin(), vec_col_sorted.end(), d_cooColInd_gpu);
    thrust::copy(vec_row_sorted.begin(), vec_row_sorted.end(), d_cooRowInd_gpu);
    thrust::copy(vec_val_sorted.begin(), vec_val_sorted.end(), d_cooVal_gpu);
  }
  // Transform sparse matrix to CSR format
  chkErrqCusparse(cusparseXcoo2csr(d_cusp_handle, d_cooRowInd_gpu, d_nnz, N, d_csrRowPtr_gpu, d_base));
  
  // Create descriptor for matrix M
  chkErrqCusparse(cusparseCreateMatDescr(&d_descr_M));
  // chkErrqCusparse(cusparseSetMatType(d_descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));
  chkErrqCusparse(cusparseSetMatType(d_descr_M, CUSPARSE_MATRIX_TYPE_SYMMETRIC));
  chkErrqCusparse(cusparseSetMatIndexBase(d_descr_M, CUSPARSE_INDEX_BASE_ZERO));
  chkErrqCusparse(cusparseSetMatFillMode(d_descr_M, CUSPARSE_FILL_MODE_UPPER)); // or LOWER
  chkErrqCusparse(cusparseSetMatDiagType(d_descr_M, CUSPARSE_DIAG_TYPE_NON_UNIT));
  // Print matrix 
  // print_csr_matrix_in_dense_format(d_cusp_handle, N, N, d_nnz, d_descr_M, d_cooVal_gpu, d_csrRowPtr_gpu, d_cooColInd_gpu, d_x);

  // Copy matrix to the CPU
  if(1){
    d_cooVal = new double [d_nnz];
    d_cooRowInd = new int [d_nnz];
    d_cooColInd = new int [d_nnz];
    d_csrRowPtr = new int [(N) + 1];
    chkErrq(cudaMemcpy(d_cooVal, d_cooVal_gpu, d_nnz * sizeof(double), cudaMemcpyDeviceToHost));
    chkErrq(cudaMemcpy(d_cooRowInd, d_cooRowInd_gpu, d_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    chkErrq(cudaMemcpy(d_cooColInd, d_cooColInd_gpu, d_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    chkErrq(cudaMemcpy(d_csrRowPtr, d_csrRowPtr_gpu, ((3 * d_number_of_blobs) + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    
    // for(int i=0; i<d_nnz; i++){
    //   if(isnan(d_cooRowInd[i]) || isnan(d_cooColInd[i]) || isnan(d_cooVal[i])){
    // 	cout << i << " --- " << d_cooRowInd[i] << "  " << d_cooColInd[i] << "  " << d_cooVal[i] << endl;
    // 	cout << "problem in d_cooRowInd[i]) || isnan(d_cooColInd[i]) || isnan(d_cooVal[i]) " << i << " --- " << d_cooRowInd[i] << "  " << d_cooColInd[i] << "  " << d_cooVal[i] << endl;
    // 	exit(1);
    //   }
    // }
    // for(int i=0; i < ((N) + 1); i++){
    //   if(isnan(d_csrRowPtr[i])){
    // 	cout << i << " --- " << d_csrRowPtr[i] << endl;
    // 	cout << "problem in d_csrRowPtr[i] " << i << "  " << d_csrRowPtr[i] << endl;
    // 	exit(1);
    //   }
    // }
    delete[] d_cooVal;
    delete[] d_cooRowInd;
    delete[] d_cooColInd;
    delete[] d_csrRowPtr;
  }

  // Create info structure for incomplete Cholesky
  // cusparseCreateCsric02Info(&d_info_M); for version 7.5
  cusparseCreateSolveAnalysisInfo(&d_info_M);
  cusparseOperation_t operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
  if(1){
    chkErrqCusparse(cusparseDcsrsv_analysis(d_cusp_handle, 
					    operation, /*CUSPARSE_OPERATION_NON_TRANSPOSE*/
					    N,
					    d_nnz,
					    d_descr_M, 
					    d_cooVal_gpu,
					    d_csrRowPtr_gpu, 
					    d_cooColInd_gpu,
					    d_info_M));
    chkErrq(cudaDeviceSynchronize());
  }

  // print_csr_matrix_in_dense_format(d_cusp_handle, N, N, d_nnz, d_descr_M, d_cooVal_gpu, d_csrRowPtr_gpu, d_cooColInd_gpu, d_x);    
  // Compute incomplete cholesky 
  if(1){
    // chkErrqCusparse(cusparseSetMatType(d_descr_M, CUSPARSE_MATRIX_TYPE_SYMMETRIC));
    // chkErrqCusparse(cusparseSetMatType(d_descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));
    chkErrqCusparse(cusparseDcsric0(d_cusp_handle,
				    operation,
				    N,
				    d_descr_M,
				    d_cooVal_gpu,
				    d_csrRowPtr_gpu,
				    d_cooColInd_gpu,
				    d_info_M));
  }
  chkErrq(cudaDeviceSynchronize());

  // Print matrix 
  // print_csr_matrix_in_dense_format(d_cusp_handle, N, N, d_nnz, d_descr_M, d_cooVal_gpu, d_csrRowPtr_gpu, d_cooColInd_gpu, d_x);    

  // Create descriptor for matrix Cholesky factor L
  chkErrqCusparse(cusparseCreateMatDescr(&d_descr_L));
  // chkErrqCusparse(cusparseSetMatType(d_descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));
  chkErrqCusparse(cusparseSetMatType(d_descr_L, CUSPARSE_MATRIX_TYPE_TRIANGULAR));
  chkErrqCusparse(cusparseSetMatIndexBase(d_descr_L, CUSPARSE_INDEX_BASE_ZERO));
  chkErrqCusparse(cusparseSetMatFillMode(d_descr_L, CUSPARSE_FILL_MODE_UPPER)); // or LOWER
  chkErrqCusparse(cusparseSetMatDiagType(d_descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT));

  // Create info to solve with L (Cholesky factor)
  cusparseCreateSolveAnalysisInfo(&d_info_L);
  chkErrqCusparse(cusparseDcsrsv_analysis(d_cusp_handle, 
					  CUSPARSE_OPERATION_NON_TRANSPOSE,
					  N,
					  d_nnz,
					  d_descr_L, 
					  d_cooVal_gpu,
					  d_csrRowPtr_gpu, 
					  d_cooColInd_gpu,
					  d_info_L));
  chkErrq(cudaDeviceSynchronize());

  // Create info to solve with L^T
  cusparseCreateSolveAnalysisInfo(&d_info_LT);
  chkErrqCusparse(cusparseDcsrsv_analysis(d_cusp_handle, 
					  CUSPARSE_OPERATION_TRANSPOSE,
					  N,
					  d_nnz,
					  d_descr_L, 
					  d_cooVal_gpu,
					  d_csrRowPtr_gpu, 
					  d_cooColInd_gpu,
					  d_info_LT));
  chkErrq(cudaDeviceSynchronize());
  d_icc_is_initialized = 1;  
  cout << "icc_init DONE" << endl;
  return 0;
}


/*
  Muliply by Cholesky factorization L.
  L*x = b
  x_gpu and solution b_gpu are on the GPU
*/
int icc::multL_gpu(const double *x_gpu, double *b_gpu, cusparseOperation_t operation){
  int N = d_number_of_blobs * 3;
  cusparseMatrixType_t mat_type = cusparseGetMatType(d_descr_L);
  double alpha = 1;
  double beta = 0;
  chkErrqCusparse(cusparseSetMatType(d_descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
  chkErrqCusparse(cusparseDcsrmv(d_cusp_handle, 
   				 operation,
   				 N,
   				 N,
   				 d_nnz,
				 &alpha,
				 d_descr_L,
				 d_cooVal_gpu,
				 d_csrRowPtr_gpu,
				 d_cooColInd_gpu,
				 x_gpu,
				 &beta,
				 b_gpu));
  chkErrq(cudaDeviceSynchronize());
  chkErrqCusparse(cusparseSetMatType(d_descr_L, mat_type));
  return 0;
}


/*
  Apply preconditioner mobility
  L^{-T} * M * L^{-1} * x = b
*/
//int icc::mult_precondM(const double *x, double *b){ 
int icc::multL(const bp::object x_obj, bp::object b_obj){ 
  // Extract pointers 
  PyObject* pobj = x_obj.ptr();
  Py_buffer pybuf;
  PyObject_GetBuffer(pobj, &pybuf, PyBUF_SIMPLE);
  void *buf = pybuf.buf;
  double *x = (double*)buf;
  PyObject* pbobj = b_obj.ptr();
  Py_buffer pybbuf;
  PyObject_GetBuffer(pbobj, &pybbuf, PyBUF_SIMPLE);
  void *bbuf = pybbuf.buf;
  double *b = (double*)bbuf;
  
  // Allocate memory
  int N = d_number_of_blobs * 3;
  double *x_gpu, *b_gpu;
  chkErrq(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
  chkErrq(cudaMalloc((void**)&b_gpu, N * sizeof(double)));
  // Copy data from CPU to GPU
  chkErrq(cudaMemcpy(x_gpu, x, N * sizeof(double), cudaMemcpyHostToDevice));
  // Compute product
  cusparseOperation_t operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
  multL_gpu(x_gpu, b_gpu, operation);
  // Copy data from GPU to CPU
  chkErrq(cudaMemcpy(b, b_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));
  // Free memory
  chkErrq(cudaFree(b_gpu));
  chkErrq(cudaFree(x_gpu));
  return 0;
}


/*
  Solve with Cholesky factor L
  L*x = b
  solution x_gpu and RHS b_gpu are in the GPU
*/
int icc::solveL_gpu(const double *b_gpu, double *x_gpu){ 
  int N = d_number_of_blobs * 3;
  double alpha = 1;
  alpha = 1;
  chkErrqCusparse(cusparseDcsrsv_solve(d_cusp_handle,
				       CUSPARSE_OPERATION_NON_TRANSPOSE,
				       N, 
				       &alpha,
				       d_descr_L,
				       d_cooVal_gpu,
				       d_csrRowPtr_gpu, 
				       d_cooColInd_gpu,
				       d_info_L,
				       b_gpu, 
				       x_gpu));
  chkErrq(cudaDeviceSynchronize());
  return 0;
}

/*
  Solve with Cholesky (transpose) factor L^T
  L^T*x = b
  solution x_gpu and RHS b_gpu are on the GPU
*/
int icc::solveLT_gpu(const double *b_gpu, double *x_gpu){
  int N = d_number_of_blobs * 3;
  double alpha = 1;
  alpha = 1;
  chkErrqCusparse(cusparseDcsrsv_solve(d_cusp_handle,
				       CUSPARSE_OPERATION_TRANSPOSE,
				       N, 
				       &alpha,
				       d_descr_L,
				       d_cooVal_gpu,
				       d_csrRowPtr_gpu, 
				       d_cooColInd_gpu,
				       d_info_LT,
				       b_gpu, 
				       x_gpu));
  chkErrq(cudaDeviceSynchronize());
  return 0;
}


/*
  Apply preconditioner mobility
  L^{-T} * M * L^{-1} * x = b
*/
int icc::mult_precondM_gpu(const double *x_gpu, double *b_gpu){
  // First, solve L*b=x
  solveL_gpu(x_gpu, d_aux_gpu);
  // Second, apply mobility M*x = b
  velocity_from_force<<<d_num_blocks, d_threads_per_block>>>(d_x_gpu,
							     d_aux_gpu,					
							     b_gpu,
							     d_number_of_blobs,
							     d_eta,
							     d_blob_radius);
  chkErrq(cudaDeviceSynchronize());
  chkErrq(cudaMemcpy(d_aux_gpu, b_gpu, 3 * d_number_of_blobs * sizeof(double), cudaMemcpyDeviceToDevice)); 
  // Third, solve L.T*b = x
  solveLT_gpu(d_aux_gpu, b_gpu);
  return 0;
}


/*
  Apply preconditioner mobility
  L^{-T} * M * L^{-1} * x = b
*/
//int icc::mult_precondM(const double *x, double *b){ 
int icc::mult_precondM(const bp::object x_obj, bp::object b_obj){ 
  // Extract pointers 
  PyObject* pobj = x_obj.ptr();
  Py_buffer pybuf;
  PyObject_GetBuffer(pobj, &pybuf, PyBUF_SIMPLE);
  void *buf = pybuf.buf;
  double *x = (double*)buf;
  PyObject* pbobj = b_obj.ptr();
  Py_buffer pybbuf;
  PyObject_GetBuffer(pbobj, &pybbuf, PyBUF_SIMPLE);
  void *bbuf = pybbuf.buf;
  double *b = (double*)bbuf;
  
  // Allocate memory
  int N = d_number_of_blobs * 3;
  double *x_gpu, *b_gpu;
  chkErrq(cudaMalloc((void**)&x_gpu, N * sizeof(double)));
  chkErrq(cudaMalloc((void**)&b_gpu, N * sizeof(double)));
  // Copy data from CPU to GPU
  chkErrq(cudaMemcpy(x_gpu, x, N * sizeof(double), cudaMemcpyHostToDevice));
  // Compute product
  mult_precondM_gpu(x_gpu, b_gpu);
  // Copy data from GPU to CPU
  chkErrq(cudaMemcpy(b, b_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));
  // Free memory
  chkErrq(cudaFree(b_gpu));
  chkErrq(cudaFree(x_gpu));
  return 0;
}


int main(){
  // Define parameters
  int status;
  double aux;
  double blob_radius = 1.0;
  double eta = 1.0;
  double cutoff = 75;
  int number_of_blobs = 1000;
  int N = number_of_blobs * 3;

  // Create CPU arrays
  double *x; 
  if(0){
    x = new double [N];
    for(int i=0; i<(N); i++){
      x[i] = 1000.0 * rand() / RAND_MAX;
    }
  }
  else{
    ifstream coor("shell_3d_Nblob_1_R_h_1.clones");
    coor >> number_of_blobs;
    N = number_of_blobs * 3;
    x = new double [N];
    for(int i=0; i<number_of_blobs; i++){
      coor >> x[i*3] >> x[i*3+1] >> x[i*3+2];
      coor >> aux >> aux >> aux >> aux;
    }
    coor.close();
  }
 
  // Create icc object
  icc icc_obj = icc(blob_radius, eta, cutoff, number_of_blobs, x);
  
  // Build sparse mobility matrix
  status = icc_obj.init_icc();
  cout << "Build sparse mobility matrix = " << status << endl;
  
  // Test solve L*x = b
  double *b = new double[N];
  for(int i=0; i<N; i++){
    b[i] = 1.0;
  }
  double *b_gpu, *x_gpu;
  chkErrq(cudaMalloc((void**)&b_gpu, N * sizeof(double)));  
  chkErrq(cudaMalloc((void**)&x_gpu, N * sizeof(double)));  
  // Move info to gpu
  chkErrq(cudaMemcpy(x_gpu, b, N * sizeof(double), cudaMemcpyHostToDevice));    
  chkErrq(cudaMemcpy(b_gpu, b, N * sizeof(double), cudaMemcpyHostToDevice));      

  // Compute RHS = L*x
  cusparseOperation_t operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
  icc_obj.multL_gpu(x_gpu, b_gpu, operation);
  chkErrq(cudaMemcpy(b, b_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));    
  for(int i=0; i<N; i++){
    if(isnan(b[i])){
      cout << "icc_obj.multL_gpu -- i, b = " << i << "    " << b[i] << endl;
      exit(1);
    }
  }
  // Solve system L*x = RHS
  icc_obj.solveL_gpu(b_gpu, x_gpu);
  chkErrq(cudaMemcpy(x, x_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));    
  chkErrq(cudaDeviceSynchronize());
  for(int i=0; i<N; i++){
    cout << "icc_obj.solveL_gpu i, x = " << i << "    " << x[i] << endl;
    b[i] = 1.0;
  }
  chkErrq(cudaMemcpy(x_gpu, b, N * sizeof(double), cudaMemcpyHostToDevice));    

  cout << endl;
  if(1){
    // Compute RHS = L.T*x
    operation = CUSPARSE_OPERATION_TRANSPOSE;
    icc_obj.multL_gpu(x_gpu, b_gpu, operation);
    chkErrq(cudaMemcpy(b, b_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));    
    for(int i=0; i<N; i++){
      cout << "icc_obj.multL_gpu TRANSPOSE i, b = " << i << "    " << b[i] << endl;
    }
    // Solve system L*x = RHS
    icc_obj.solveLT_gpu(b_gpu, x_gpu);
    chkErrq(cudaMemcpy(x, x_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));    
    chkErrq(cudaDeviceSynchronize());
    for(int i=0; i<N; i++){
      cout << "icc_obj.solveLT_gpu TRANSPOSE i, x = " << i << "    " << x[i] << endl;
      b[i] = 1.0;
    }
    chkErrq(cudaMemcpy(x_gpu, b, N * sizeof(double), cudaMemcpyHostToDevice));    
  }
  cout << endl;

  if(1){
    icc_obj.mult_precondM_gpu(x_gpu, b_gpu);
    chkErrq(cudaMemcpy(b, b_gpu, N * sizeof(double), cudaMemcpyDeviceToHost));    
    for(int i=0; i<N; i++){
      cout << "icc_obj.mult_precondM_gpu i, b = " << i << "    " << b[i] << endl;
    }
  }

  


  // Free GPU memory
  chkErrq(cudaFree(x_gpu));
  chkErrq(cudaFree(b_gpu));
  // Free CPU memory
  delete[] b;
  cout << "before x" << endl;
  delete[] x;
  cout << "# End" << endl;
  return 0;
}




BOOST_PYTHON_MODULE(icc_ext)
{
  using namespace boost::python;
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  class_<icc>("icc", init<const double, const double, const double, const int, const double*>())
    .def(init<const double, const double, const double, const int, bp::object>())
    .def("init_icc", &icc::init_icc)
    .def("mult_precondM_gpu", &icc::mult_precondM_gpu)
    .def("mult_precondM", &icc::mult_precondM)
    .def("multL", &icc::multL)
    ;
}
