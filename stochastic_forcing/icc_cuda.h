#include <iostream>
#include <stdio.h>
#include "cusparse_v2.h"
#include <boost/python.hpp>

using namespace std;
namespace bp = boost::python;

class icc{
public:
  /*
    Constructor: build the sparse mobility matrix M
    and compute the Cholesky factorization M=L*L.T
    where L is a lower triangular matrix.
  */
  icc(const double blob_radius, 
      const double eta, 
      const double cutoff,
      const int number_of_blobs,
      const double *x);

  /*
    Constructor: build the sparse mobility matrix M
    and compute the Cholesky factorization M=L*L.T
    where L is a lower triangular matrix.
  */
  icc(const double blob_radius, 
      const double eta, 
      const double cutoff,
      const int number_of_blobs,
      bp::object x_obj);

  /*
    Destructor: free memory on the GPU and CPU.
  */
  ~icc();

  /*
    Build sparse mobility matrix M.
  */
  int init_icc();

  /*
    Muliply by Cholesky factorization L.
    L*x = b
    x_gpu and solution b_gpu are on the GPU
  */
  int multL_gpu(const double *x_gpu, double *b_gpu, const cusparseOperation_t operation);

  /*
    Muliply by Cholesky factorization L.
    L*x = b
    x_gpu and solution b_gpu are on the GPU
  */
  int multL(const bp::object x_obj, bp::object b_obj);

  /*
    Solve with Cholesky factor L
    L*x = b
    solution x_gpu and RHS b_gpu are on the GPU
  */
  int solveL_gpu(const double *b_gpu, double *x_gpu);

  /*
    Solve with Cholesky (transpose) factor L^T
    L^T*x = b
    solution x_gpu and RHS b_gpu are on the GPU
  */
  int solveLT_gpu(const double *b_gpu, double *x_gpu);

  /*
    Apply preconditioner mobility
    L^{-T} * M * L^{-1} * x = b
   */
  int mult_precondM_gpu(const double *x_gpu, double *b_gpu);

  /*
    Apply preconditioner mobility
    L^{-T} * M * L^{-1} * x = b
   */
  int mult_precondM(const bp::object x_obj, bp::object b_obj);

private:
  // CPU variables
  int d_icc_is_initialized;
  double d_blob_radius, d_eta, d_cutoff;
  int d_number_of_blobs;
  const double *d_x;
  int d_threads_per_block, d_num_blocks;
  unsigned long long int d_nnz;
  cusparseHandle_t d_cusp_handle;
  cusparseIndexBase_t d_base;
  cusparseStatus_t d_cusp_status;
  double *d_cooVal;
  int *d_cooRowInd, *d_cooColInd, *d_csrRowPtr;
  cusparseMatDescr_t d_descr_M, d_descr_L;
  // csric02Info_t d_info_M; for version cuda 7.5
  cusparseSolveAnalysisInfo_t d_info_M, d_info_L, d_info_LT;

  // GPU variables
  double *d_x_gpu, *d_aux_gpu;
  unsigned long long int *d_nnz_gpu;
  double *d_cooVal_gpu;
  int *d_cooRowInd_gpu, *d_cooColInd_gpu, *d_csrRowPtr_gpu, *d_cooVal_sorted_gpu;
  unsigned long long int *d_index_gpu;
};


