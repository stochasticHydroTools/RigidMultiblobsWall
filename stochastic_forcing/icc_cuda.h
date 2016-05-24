#include <iostream>
using namespace std;

class icc{
public:
  /*
    Constructor: build the sparse mobility matrix M
    and compute the Cholesky factorization M=L*L.T
    where L is a lower triangular matrix.
  */
  icc();

  /*
    Destructor: free memory on the GPU and CPU.
  */
  ~icc();

  /*
    Muliply by Cholesky factorization L.
  */
  int mult_L();

  /*
    Muliply by inverse of the Cholesky factorization L^{-1}.
  */
  int mult_invL();

private:
  int d_icc_is_initialized;
  double d_blob_radius, d_eta;
  int d_number_of_blobs;
};


