// Functions for tetrahedron simulation written
// in C++ for improved speed.

#include <boost/python.hpp>
#include <math.h>
#include <vector>
#include <iostream>

void TestList(boost::python::list list_to_print) {
    // Test using the python list implemented in boost.
    int n  = boost::python::len(list_to_print);
    for (int k = 0; k < n; ++k) {
        std::cout << "x at " << k << " is "
                  << boost::python::extract<double>(list_to_print[k])
                  << std::endl;
    }
}

double* SingleWallFluidMobility(double** r_vectors, double eta,
                                double a, int num_particles) {
	// Create the mobility of particles in a fluid with a single wall at z = 0.
	double* R;
	R = new double[3];
	double h = 1.0;
	for (int j = 0; j < num_particles; ++j) {
		for (int k = 0; k < num_particles; ++k) {
			// Here notation is based on appendix C of the Swan and Brady paper:
			//  'Simulation of hydrodynamically interacting particles near a no-slip
			//   boundary.'
			double h = r_vectors[k][2];
			for (int l = 0; l < 2; ++l) {
				R[l] = (r_vectors[j][l] - (r_vectors[k][l]))/a;
			}
			R[2] = (r_vectors[j][2] - (r_vectors[k][2]) - 2.0*h)/a;

		}
	}
	return R;
  //     R_norm = np.linalg.norm(R)
  //     e = R/R_norm
  //     e_3 = np.array([0., 0., e[2]])
  //     h_hat = h/(a*R[2])
  //     # Taken from Appendix C expression for M_UF
  //     mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] += (1./(6.*np.pi*eta*a))*(
  //       -0.25*(3.*(1. - 6.*h_hat*(1. - h_hat)*e[2]**2)/R_norm
  //              - 6.*(1. - 5.*e[2]**2)/(R_norm**3)
  //              + 10.*(1. - 7.*e[2]**2)/(R_norm**5))*np.outer(e, e)
  //        - (0.25*(3.*(1. + 2.*h_hat*(1. - h_hat)*e[2]**2)/R_norm
  //                 + 2.*(1. - 3.*e[2]**2)/(R_norm**3)
  //                 - 2.*(2. - 5.*e[2]**2)/(R_norm**5)))*np.identity(3)
  //        + 0.5*(3.*h_hat*(1. - 6.*(1. - h_hat)*e[2]**2)/R_norm
  //               - 6.*(1. - 5.*e[2]**2)/(R_norm**3)
  //               + 10.*(2. - 7.*e[2]**2)/(R_norm**5))*np.outer(e, e_3)
  //        + 0.5*(3.*h_hat/R_norm - 10./(R_norm**5))*np.outer(e_3, e)
  //        - (3.*(h_hat**2)*(e[2]**2)/R_norm 
  //           + 3.*(e[2]**2)/(R_norm**3)
  //           + (2. - 15.*e[2]**2)/(R_norm**5))*np.outer(e_3, e_3)/(e[2]**2))
      
  //     mobility[(k*3):(k*3 + 3), (j*3):(j*3 + 3)] = (
  //       mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)].T)

  // for j in range(len(r_vectors)):
  //   # Diagonal blocks, self mobility.
  //   h = r_vectors[j][2]/a
  //   for l in range(3):
  //     for m in range(3):
  //       mobility[j*3 + l][j*3 + m] += (1./(6.*np.pi*eta*a))*(
  //         (l == m)*(l != 2)*(-1./16.)*(9./h - 2./(h**3) + 1./(h**5))
  //         + (l == m)*(l == 2)*(-1./8.)*(9./h - 4./(h**3) + 1./(h**5)))
  // return mobility
}


std::vector<double> TestArray(double a) {
	std::vector<double> R;
	R.push_back(0.0);
	R.push_back(5.0);
	R.push_back(a);
	R.push_back(2.0);
	R.push_back(1.0);
	return R;
}


BOOST_PYTHON_MODULE(tetrahedron_ext)
{
    using namespace boost::python;
    def("test_list", TestList);
    def("test_array", TestArray);
}
