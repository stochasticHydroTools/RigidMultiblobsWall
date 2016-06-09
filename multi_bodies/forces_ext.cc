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

  In this example the force is derived from a Yukawa potential
  
  U = eps * exp(-r_norm / b) / r_norm
  
  with
  eps = potential strength
  r_norm = distance between blobs
  b = Debye length
 */
void blobBlobForce(double *r,
		   double *f,
		   double eps,
		   double b){
  double r_norm = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
  double f_scalar = -((eps / b) + (eps / r_norm)) * exp(-r_norm / b) / (r_norm*r_norm);
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
			int number_of_blobs){

  
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
      }
      // Compute force between blobs i and j
      blobBlobForce(r, f, repulsion_strength, debye_length);
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
