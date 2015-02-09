// Functions for tetrahedron simulation written
// in C++ for improved speed.

#include <boost/python.hpp>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>

namespace bp = boost::python;

// Test functions to play with passing data to/from python.
void PrintTest(void) {
  std::cout << "printing stuff" << std::endl;
}

void TestList(bp::list list_of_lists) {
  // Test using the python list implemented in boost.
  bp::list list_to_print = bp::extract<bp::list>(list_of_lists[0]);
  int n  = bp::len(list_to_print);
  for (int k = 0; k < n; ++k) {
    std::cout << "x at " << k << " is "
              << bp::extract<double>(list_to_print[k])
              << std::endl;
  }
}

void SingleWallFluidMobility(bp::list r_vectors, 
                             double eta,
                             double a, int num_particles,
                             bp::numeric::array mobility) {
  double pi = 3.1415926535897932;
  // Create the mobility of particles in a fluid with a single wall at z = 0.
  double* R;
  R = new double[3];
  double h = 0.0;
  for (int j = 0; j < num_particles; ++j) {
    for (int k = j+1; k < num_particles; ++k) {
      // Here notation is based on appendix C of the Swan and Brady paper:
      //  'Simulation of hydrodynamically interacting particles near a no-slip
      //   boundary.'
      bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(r_vectors[j]);
      bp::numeric::array  r_vector_2 = bp::extract<bp::numeric::array>(r_vectors[k]);
      h = bp::extract<double>(r_vector_2[2]);
      for (int l = 0; l < 2; ++l) {
        R[l] = (bp::extract<double>(r_vector_1[l]) -
                bp::extract<double>(r_vector_2[l]))/a;
      }
      R[2] = (bp::extract<double>(r_vector_1[2]) -
              bp::extract<double>(r_vector_2[2]) + 2.0*h)/a;
      double R_norm = 0.0;
      for (int l = 0; l < 3; ++l) {
        R_norm += R[l]*R[l];
      }
      R_norm = sqrt(R_norm);
      double* e = new double[3];
      for (int l = 0; l < 3; ++l) {
        e[l] = R[l]/R_norm;
      }
      double* e_3 = new double[3];
      e_3[0] = 0.0;
      e_3[1] = 0.0;
      e_3[2] = e[2];
      
      double h_hat = h/(a*R[2]);
      // Taken from Appendix C expression for M_UF.
      for (int l = 0; l < 3; ++l) {
        bp::numeric::array current_row =
            bp::extract<bp::numeric::array>(mobility[j*3 + l]);
        for (int m = 0; m < 3; ++m) {
          current_row[k*3 + m] +=
              (1.0/(6.0*pi*eta*a))*
              (-0.25*(3.0*(1.0 - 6.0*h_hat*(1. - h_hat)*pow(e[2],2))/R_norm
                      - 6.0*(1.0 - 5.0*pow(e[2],2))/(pow(R_norm,3))
                      + 10.0*(1.0 - 7.0*pow(e[2],2))/(pow(R_norm,5)))*(e[l]*e[m])
               - (0.25*(3.0*(1.0 + 2.0*h_hat*(1. - h_hat)*pow(e[2],2))/R_norm
                        + 2.0*(1.0 - 3.0*pow(e[2],2))/(pow(R_norm,3))
                        - 2.0*(2.0 - 5.0*pow(e[2],2))/(pow(R_norm,5))))*(l == m ? 1.0 : 0.0)
               + 0.5*(3.0*h_hat*(1. - 6.0*(1. - h_hat)*pow(e[2],2))/R_norm
                      - 6.0*(1.0 - 5.0*pow(e[2],2))/(pow(R_norm,3))
                      + 10.0*(2.0 - 7.0*pow(e[2],2))/(pow(R_norm,5)))*(e[l]*e_3[m])
               + 0.5*(3.0*h_hat/R_norm - 10./(pow(R_norm,5)))*(e_3[l]*e[m])
               - (3.0*(pow(h_hat,2))*(pow(e[2],2))/R_norm 
                  + 3.0*(pow(e[2],2))/(pow(R_norm,3))
                  + (2.0 - 15.0*pow(e[2],2))/(pow(R_norm,5)))*
               (e_3[l]*e_3[m])/(pow(e[2],2)));
        }
      }

      for (int m = 0; m < 3; ++m) {
        bp::numeric::array current_row =
            bp::extract<bp::numeric::array>(mobility[k*3 + m]);
        for (int l = 0; l < 3; ++l) {
          current_row[j*3 + l] = mobility[j*3 + l][k*3 + m];
        }
      }
    }
  }
  
  // Diagonal blocks, self mobility.
  for (int j = 0; j < num_particles; ++j) {
    bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(r_vectors[j]);
    h = bp::extract<double>(r_vector_1[2])/a;
    for (int l = 0; l < 3; ++l) {
      bp::numeric::array current_row =
          bp::extract<bp::numeric::array>(mobility[j*3 + l]);
      for (int m = 0; m < 3; ++m) {
        current_row[j*3 + m] += (1./(6.0*pi*eta*a))*(
            (l == m ? 1.0 : 0.0)*(l != 2 ? 1.0 : 0.0)*(-1./16.)*
            (9./h - 2./(pow(h,3)) + 1./(pow(h,5)))
            + (l == m ? 1.0 : 0.0 )*(l == 2 ? 1.0 : 0.0)*(-1./8.)*
            (9./h - 4./(pow(h,3)) + 1./(pow(h,5))));
      }
    }
  }
}

void ConstructRotationMatrix(bp::list orientation, double* rotation_matrix) {
	// Construct rotation matrix from a list of quaternion entries.
	// orientation = [s, p1, p2, p3]
	// rotation_matrix is output as a pointer to 9 doubles, stored in row 
	// major order.
	double s = bp::extract<double>(orientation[0]);
	double* p = new double[3];
	p[0] = bp::extract<double>(orientation[1]);
	p[1] = bp::extract<double>(orientation[2]);
	p[2] = bp::extract<double>(orientation[3]);
	// add 2 * p p^t
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			rotation_matrix[i*3 + j] = 2.0*p[i]*p[j];
		}
	}
	// Add 2.*(s^2 - 0.5)*identity
	for (int i = 0; i < 3; ++i) {
		rotation_matrix[i*3 + i] += 2.0*s*s - 1.0;
	}

	// Add 2.* s cross P.
	rotation_matrix[1] += -2.0*p[2]*s;
	rotation_matrix[2] += 2.0*p[1]*s;
	rotation_matrix[3] += 2.0*p[2]*s;
	rotation_matrix[5] += -2.0*p[0]*s;
	rotation_matrix[6] += -2.0*p[1]*s;
	rotation_matrix[7] += 2.0*p[0]*s;
}


void GetFreeRVectors(bp::numeric::array location, bp::list orientation,
								 bp::list r_vectors) {
	// Get R vectors from the location and orientation of a free tetrahedron.
	// location is a list of coordinates, x, y, z.
	// orientation is a list of entries of a Quaternion, s, p1, p2, p3.
	// output is r_vectors, a list of arrays of r_vectors.
	// See get_free_r_vectors in tetrahedron_free.py for 
	// a description of the geometric setup.
	
	// Rotation matrix is a 3x3 matrix stored in row major order.
	double* rotation_matrix = new double[9];
	// Initial configuration, each 3 entries is one initial r vector.
	double* initial_r = new double[9];
	initial_r[0] = 0.0;
	initial_r[1] = 2.0/sqrt(3.0);
	initial_r[2] = -2.0*sqrt(2.0)/sqrt(3.0);
	initial_r[3] = -1.0;
	initial_r[4] = -1.0/sqrt(3.0);
	initial_r[5] = -2*sqrt(2.0)/sqrt(3.0);
	initial_r[6] = 1.0;
	initial_r[7] = -1.0/sqrt(3.0);
	initial_r[8] = -2*sqrt(2.0)/sqrt(3.0);
	ConstructRotationMatrix(orientation, rotation_matrix);

	for (int k = 0; k < 3; ++k) {
		bp::numeric::array current_r_vector =
			bp::extract<bp::numeric::array>(r_vectors[k]);
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				current_r_vector[i] += rotation_matrix[i*3 + j]*initial_r[k*3 + j];
			}
		}
		// Add location.
		for(int i = 0; i < 3; ++i) {
			current_r_vector[i] += location[i];
		}
	}
}


BOOST_PYTHON_MODULE(tetrahedron_ext)
{
  using namespace boost::python;
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  def("print_test", PrintTest);
  def("test_list", TestList);
	//  def("single_wall_fluid_mobility", SingleWallFluidMobility);
  def("get_free_r_vectors", GetFreeRVectors);
}

