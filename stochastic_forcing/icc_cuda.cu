#include "icc_cuda.h"

/*
  Constructor: build the sparse mobility matrix M
  and compute the Cholesky factorization M=L*L.T
  where L is a lower triangular matrix.
*/
icc::icc(){

}

/*
  Destructor: free memory on the GPU and CPU.
*/
icc::~icc(){

}




int main(){

  // Define parameters
  double blob_radius = 1.0;
  double eta = 1.0;
  int number_of_blobs = 2;
  
  // Create icc object
  icc icc_obj = icc();

  cout << "# End" << endl;
  return 0;
}
