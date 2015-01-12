// Functions for tetrahedron simulation written
// in C++ for improved speed.

#include <boost/python.hpp>
#include <iostream>
#include <math.h>
#include <numpy/ndarrayobject.h>
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

BOOST_PYTHON_MODULE(tetrahedron_ext)
{
  using namespace boost::python;
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  def("print_test", PrintTest);
  def("test_list", TestList);
  def("single_wall_fluid_mobility", SingleWallFluidMobility);
}

