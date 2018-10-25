// Functions for fluid mobilities written
// in C++ for improved speed.

#include <boost/python.hpp>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>

namespace bp = boost::python;

/*
  This function compute the force between two blobs
  with vector between blob centers r.

  In this example the force is derived from the potential
  
  U(r) = U0 + U0 * (2*a-r)/b   if z<2*a
  U(r) = U0 * exp(-(r-2*a)/b)  iz z>=2*a
  
  with
  eps = potential strength
  r_norm = distance between blobs
  b = Debye length
  a = blob_radius
 */
void blobBlobForce(double *r,
		   double *f,
		   double eps,
		   double b,
                   double a){
  double r_norm = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
  double f_scalar; 
  if(r_norm > 2*a){
    f_scalar = -(eps / b) * exp(-(r_norm-2*a) / b) / r_norm; 
  }
  else if(r_norm > 0){
    f_scalar = -(eps / b) / r_norm;
  }
  f[0] = f_scalar * r[0];
  f[1] = f_scalar * r[1];
  f[2] = f_scalar * r[2];
}
		   

/*
  This function computes the blob-blob forces for
  all blobs.
 */
void calcBlobBlobForces(bp::numeric::array r_vectors,
			bp::numeric::array force,
			double repulsion_strength,
			double debye_length,
			double blob_radius,
			int number_of_blobs,
			bp::numeric::array periodic_length){

  bp::numeric::array L = bp::extract<bp::numeric::array>(periodic_length);  
  double r[3];
  double f[3];

  // Set initial force to zero
  for(int blob_i=0; blob_i<number_of_blobs; blob_i++){
    for (int l = 0; l<3; l++){
      force[blob_i * 3 + l] = 0;
    }
  }

  // Double loop over blobs
  for(int blob_i=0; blob_i<number_of_blobs-1; blob_i++){
    bp::numeric::array r_vector_i = bp::extract<bp::numeric::array>(r_vectors[blob_i]);
    for(int blob_j=blob_i+1; blob_j<number_of_blobs; blob_j++){
      bp::numeric::array r_vector_j = bp::extract<bp::numeric::array>(r_vectors[blob_j]);
      
      // Compute vector from blob i to blob j
      for (int l = 0; l<3; l++){
        r[l] = (bp::extract<double>(r_vector_j[l]) - bp::extract<double>(r_vector_i[l]));
	
	// Project a vector r to the minimal image representation
	// centered around (0,0,0) and of size L=(Lx, Ly, Lz). If 
	// any dimension of L is equal or smaller than zero the 
	// box is assumed to be infinite in that direction.
	if(bp::extract<double>(L[l]) > 0){
	  r[l] = r[l] - int(r[l] / bp::extract<double>(L[l]) + 0.5 * (int(r[l]>0) - int(r[l]<0))) * bp::extract<double>(L[l]);
	}
      }
      // Compute force between blobs i and j
      blobBlobForce(r, f, repulsion_strength, debye_length, blob_radius);
      for (int l = 0; l<3; l++){
	force[blob_i * 3 + l] += f[l];
	force[blob_j * 3 + l] -= f[l];
      }
    }
  }
}

BOOST_PYTHON_MODULE(forces_ext)
{
  using namespace boost::python;
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  def("calc_blob_blob_forces", calcBlobBlobForces);
}
