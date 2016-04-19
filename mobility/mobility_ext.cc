// Functions for fluid mobilities written
// in C++ for improved speed.

#include <boost/python.hpp>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>

namespace bp = boost::python;



void MobilityVectorProduct_optimized(bp::list r_vectors, 
			   double eta,
			   double a, int num_particles,
                           bp::numeric::array vector,
                           bp::numeric::array vector_res ) {
  // Create the mobility of particles in a fluid with a single wall at z = 0.
  double pi = 3.1415926535897932;
  double C1, C2;
  double Mlm;
  double* R;
  double* Rim;
  R = new double[3];
  Rim = new double[3];
  double h = 0.0;
  for (int j = 0; j < num_particles; ++j) {
    bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(r_vectors[j]);

    for (int k = j+1; k < num_particles; ++k) {
      // Here notation is based on appendix C of the Swan and Brady paper:
      //  'Simulation of hydrodynamically interacting particles near a no-slip
      //   boundary.'
      bp::numeric::array  r_vector_2 = bp::extract<bp::numeric::array>(r_vectors[k]);
      h = bp::extract<double>(r_vector_2[2]);
      for (int l = 0; l < 3; ++l) {
        R[l] = (bp::extract<double>(r_vector_1[l]) -
                bp::extract<double>(r_vector_2[l]))/a;
      }
      for (int l = 0; l < 2; ++l) {
        Rim[l] = (bp::extract<double>(r_vector_1[l]) -
                bp::extract<double>(r_vector_2[l]))/a;
      }
      Rim[2] = (bp::extract<double>(r_vector_1[2]) -
              bp::extract<double>(r_vector_2[2]) + 2.0*h)/a;
              
      double R_norm = 0.0;
      double Rim_norm = 0.0;
      for (int l = 0; l < 3; ++l) {
        R_norm += R[l]*R[l];
        Rim_norm += Rim[l]*Rim[l];
      }
      R_norm = sqrt(R_norm);

      Rim_norm = sqrt(Rim_norm);
      double* e = new double[3];
      for (int l = 0; l < 3; ++l) {
        e[l] = Rim[l]/Rim_norm;
      }
      double* e_3 = new double[3];
      e_3[0] = 0.0;
      e_3[1] = 0.0;
      e_3[2] = e[2];
      
      double h_hat = h/(a*Rim[2]);
      
      if (R_norm > 2.0) {
		  C1 = 3./(4.*R_norm) + 1./(2.*pow(R_norm,3));
		  C2 = 3./(4.*R_norm) - 3./(2.*pow(R_norm,3));
	  }
	  else if (R_norm <= 2.0) {
		  C1 = 1. - 9./32.*R_norm;
		  C2 = 3./32.*R_norm;
	  }

      for (int l = 0; l < 3; ++l) {
        for (int m = 0; m < 3; ++m) {	
      // Usual RPY Tensor
	  Mlm = (l == m ? 1.0 : 0.0)*C1 + R[l]*R[m]/pow(R_norm,2)*C2; 
      // Taken from Appendix C expression for M_UF.
          Mlm +=
              (-0.25*(3.0*(1.0 - 6.0*h_hat*(1. - h_hat)*pow(e[2],2))/Rim_norm
                      - 6.0*(1.0 - 5.0*pow(e[2],2))/(pow(Rim_norm,3))
                      + 10.0*(1.0 - 7.0*pow(e[2],2))/(pow(Rim_norm,5)))*(e[l]*e[m])
               - (0.25*(3.0*(1.0 + 2.0*h_hat*(1. - h_hat)*pow(e[2],2))/Rim_norm
                        + 2.0*(1.0 - 3.0*pow(e[2],2))/(pow(Rim_norm,3))
                        - 2.0*(1.0 - 5.0*pow(e[2],2))/(pow(Rim_norm,5))))*(l == m ? 1.0 : 0.0)
               + 0.5*(3.0*h_hat*(1. - 6.0*(1. - h_hat)*pow(e[2],2))/Rim_norm
                      - 6.0*(1.0 - 5.0*pow(e[2],2))/(pow(Rim_norm,3))
                      + 10.0*(2.0 - 7.0*pow(e[2],2))/(pow(Rim_norm,5)))*(e[l]*e_3[m])
               + 0.5*(3.0*h_hat/Rim_norm - 10./(pow(Rim_norm,5)))*(e_3[l]*e[m])
               - (3.0*(pow(h_hat,2))*(pow(e[2],2))/Rim_norm 
                  + 3.0*(pow(e[2],2))/(pow(Rim_norm,3))
                  + (2.0 - 15.0*pow(e[2],2))/(pow(Rim_norm,5)))*
               (e_3[l]*e_3[m])/(pow(e[2],2)));
               
          Mlm = Mlm*(1.0/(6.0*pi*eta*a));
          
          vector_res[j*3+l] += Mlm*bp::extract<double>(vector[k*3+m]);

          // Use the fact that M is symmetric
          vector_res[k*3+m] += Mlm*bp::extract<double>(vector[j*3+l]); 
        }
      }
    }
  }
  
  // Diagonal blocks, self mobility.
  for (int j = 0; j < num_particles; ++j) {
    bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(r_vectors[j]);
    h = bp::extract<double>(r_vector_1[2])/a;
    for (int l = 0; l < 3; ++l) {
      for (int m = 0; m < 3; ++m) {
	    // RPY
	    Mlm = (l == m ? 1.0 : 0.0);
	    // Wall
            Mlm += (
            (l == m ? 1.0 : 0.0)*(l != 2 ? 1.0 : 0.0)*(-1./16.)*
            (9./h - 2./(pow(h,3)) + 1./(pow(h,5)))
            + (l == m ? 1.0 : 0.0 )*(l == 2 ? 1.0 : 0.0)*(-1./8.)*
            (9./h - 4./(pow(h,3)) + 1./(pow(h,5))));
            Mlm = Mlm*(1./(6.0*pi*eta*a));
            vector_res[j*3+l]+= Mlm*bp::extract<double>(vector[j*3+m]);
      }
    }
  }

}

//////////////////////////////////////////////////////////////////////////////////////////
/////////////// MOBILITY VECTOR PRODUCT TO BE OPTIMIZED /////////////////
//////////////////////////////////////////////////////////////////////////////////////////
void MobilityVectorProduct(bp::list r_vectors, 
			   double eta,
			   double a, int num_particles,
                           bp::numeric::array vector,
                           bp::numeric::array vector_res ) {
  // Create the mobility of particles in a fluid with a single wall at z = 0.
  double pi = 3.1415926535897932;
  double C1, C2;
  double Mlm;
  double* R;
  double* Rim;
  R = new double[3];
  Rim = new double[3];
  double h = 0.0;
  for (int j = 0; j < num_particles; ++j) {
    bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(r_vectors[j]);
    for (int k = j+1; k < num_particles; ++k) {
      // Here notation is based on appendix C of the Swan and Brady paper:
      //  'Simulation of hydrodynamically interacting particles near a no-slip
      //   boundary.'
      bp::numeric::array  r_vector_2 = bp::extract<bp::numeric::array>(r_vectors[k]);
      h = bp::extract<double>(r_vector_2[2]);
      for (int l = 0; l < 3; ++l) {
        R[l] = (bp::extract<double>(r_vector_1[l]) -
                bp::extract<double>(r_vector_2[l]))/a;
      }
      for (int l = 0; l < 2; ++l) {
        Rim[l] = (bp::extract<double>(r_vector_1[l]) -
                bp::extract<double>(r_vector_2[l]))/a;
      }
      Rim[2] = (bp::extract<double>(r_vector_1[2]) -
              bp::extract<double>(r_vector_2[2]) + 2.0*h)/a;
              
      double R_norm = 0.0;
      double Rim_norm = 0.0;
      for (int l = 0; l < 3; ++l) {
        R_norm += R[l]*R[l];
        Rim_norm += Rim[l]*Rim[l];
      }
      R_norm = sqrt(R_norm);

      Rim_norm = sqrt(Rim_norm);
      double* e = new double[3];
      for (int l = 0; l < 3; ++l) {
        e[l] = Rim[l]/Rim_norm;
      }
      double* e_3 = new double[3];
      e_3[0] = 0.0;
      e_3[1] = 0.0;
      e_3[2] = e[2];
      
      double h_hat = h/(a*Rim[2]);
      
      if (R_norm > 2.0) {
		  C1 = 3./(4.*R_norm) + 1./(2.*pow(R_norm,3));
		  C2 = 3./(4.*R_norm) - 3./(2.*pow(R_norm,3));
	  }
	  else if (R_norm <= 2.0) {
		  C1 = 1. - 9./32.*R_norm;
		  C2 = 3./32.*R_norm;
	  }

      for (int l = 0; l < 3; ++l) {
        for (int m = 0; m < 3; ++m) {	
      // Usual RPY Tensor
	  Mlm = (l == m ? 1.0 : 0.0)*C1 + R[l]*R[m]/pow(R_norm,2)*C2; 
      // Taken from Appendix C expression for M_UF.
          Mlm +=
              (-0.25*(3.0*(1.0 - 6.0*h_hat*(1. - h_hat)*pow(e[2],2))/Rim_norm
                      - 6.0*(1.0 - 5.0*pow(e[2],2))/(pow(Rim_norm,3))
                      + 10.0*(1.0 - 7.0*pow(e[2],2))/(pow(Rim_norm,5)))*(e[l]*e[m])
               - (0.25*(3.0*(1.0 + 2.0*h_hat*(1. - h_hat)*pow(e[2],2))/Rim_norm
                        + 2.0*(1.0 - 3.0*pow(e[2],2))/(pow(Rim_norm,3))
                        - 2.0*(1.0 - 5.0*pow(e[2],2))/(pow(Rim_norm,5))))*(l == m ? 1.0 : 0.0)
               + 0.5*(3.0*h_hat*(1. - 6.0*(1. - h_hat)*pow(e[2],2))/Rim_norm
                      - 6.0*(1.0 - 5.0*pow(e[2],2))/(pow(Rim_norm,3))
                      + 10.0*(2.0 - 7.0*pow(e[2],2))/(pow(Rim_norm,5)))*(e[l]*e_3[m])
               + 0.5*(3.0*h_hat/Rim_norm - 10./(pow(Rim_norm,5)))*(e_3[l]*e[m])
               - (3.0*(pow(h_hat,2))*(pow(e[2],2))/Rim_norm 
                  + 3.0*(pow(e[2],2))/(pow(Rim_norm,3))
                  + (2.0 - 15.0*pow(e[2],2))/(pow(Rim_norm,5)))*
               (e_3[l]*e_3[m])/(pow(e[2],2)));
               
          Mlm = Mlm*(1.0/(6.0*pi*eta*a));
          vector_res[j*3+l] += Mlm*bp::extract<double>(vector[k*3+m]);

          // Use the fact that M is symmetric
          vector_res[k*3+m] += Mlm*bp::extract<double>(vector[j*3+l]); 
	  
        }
      }
    }
  }
  
  // Diagonal blocks, self mobility.
  for (int j = 0; j < num_particles; ++j) {
    bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(r_vectors[j]);
    h = bp::extract<double>(r_vector_1[2])/a;
    for (int l = 0; l < 3; ++l) {
      for (int m = 0; m < 3; ++m) {
	    // RPY
	    Mlm = (l == m ? 1.0 : 0.0);
	    // Wall
            Mlm += (
            (l == m ? 1.0 : 0.0)*(l != 2 ? 1.0 : 0.0)*(-1./16.)*
            (9./h - 2./(pow(h,3)) + 1./(pow(h,5)))
            + (l == m ? 1.0 : 0.0 )*(l == 2 ? 1.0 : 0.0)*(-1./8.)*
            (9./h - 4./(pow(h,3)) + 1./(pow(h,5))));
            Mlm = Mlm*(1./(6.0*pi*eta*a));
            vector_res[j*3+l]+= Mlm*bp::extract<double>(vector[j*3+m]);

      }
    }
  }

}
//////////////////////////////////////////////////////////////////////////////////////////
/////////////// END OF MOBILITY VECTOR PRODUCT TO BE OPTIMIZED /////////////////
//////////////////////////////////////////////////////////////////////////////////////////


void SingleWallFluidMobilityCorrection(bp::list r_vectors, 
                             double eta,
                             double a, int num_particles,
                             bp::numeric::array mobility) {
  double pi = 3.1415926535897932;
  // Create the mobility of particles in a fluid with a single wall at z = 0.
  double* R;
  R = new double[3];
  
  double h = 0.0;
  for (int j = 0; j < num_particles; ++j) {
    bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(r_vectors[j]);
    
    for (int k = j+1; k < num_particles; ++k) {
      // Here notation is based on appendix C of the Swan and Brady paper:
      //  'Simulation of hydrodynamically interacting particles near a no-slip
      //   boundary.'
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
                        - 2.0*(1.0 - 5.0*pow(e[2],2))/(pow(R_norm,5))))*(l == m ? 1.0 : 0.0)
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


void RPYSingleWallFluidMobility(/*bp::list r_vectors,*/
				bp::numeric::array r_vectors,
				double eta,
				double a, int num_particles,
				bp::numeric::array mobility) {
  // Create the mobility of particles in a fluid with a single wall at z = 0.
  double pi = 3.1415926535897932;
  double C1, C2;
  double* R;
  double* Rim;
  R = new double[3];
  Rim = new double[3];
  double h = 0.0;
  for (int j = 0; j < num_particles; ++j) {
    bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(r_vectors[j]);
    for (int k = j+1; k < num_particles; ++k) {
      // Here notation is based on appendix C of the Swan and Brady paper:
      // 'Simulation of hydrodynamically interacting particles near a no-slip
      // boundary.'
      bp::numeric::array  r_vector_2 = bp::extract<bp::numeric::array>(r_vectors[k]);
      h = bp::extract<double>(r_vector_2[2]);
      for (int l = 0; l < 3; ++l) {
        R[l] = (bp::extract<double>(r_vector_1[l]) -
                bp::extract<double>(r_vector_2[l]))/a;
      }
      for (int l = 0; l < 2; ++l) {
        Rim[l] = (bp::extract<double>(r_vector_1[l]) -
                bp::extract<double>(r_vector_2[l]))/a;
      }
      Rim[2] = (bp::extract<double>(r_vector_1[2]) -
              bp::extract<double>(r_vector_2[2]) + 2.0*h)/a;
              
      double R_norm = 0.0;
      double Rim_norm = 0.0;
      for (int l = 0; l < 3; ++l) {
        R_norm += R[l]*R[l];
        Rim_norm += Rim[l]*Rim[l];
      }
      R_norm = sqrt(R_norm);

      Rim_norm = sqrt(Rim_norm);
      double* e = new double[3];
      for (int l = 0; l < 3; ++l) {
        e[l] = Rim[l]/Rim_norm;
      }
      double* e_3 = new double[3];
      e_3[0] = 0.0;
      e_3[1] = 0.0;
      e_3[2] = e[2];
      
      double h_hat = h/(a*Rim[2]);
      
      if (R_norm > 2.0) {
		  C1 = 3./(4.*R_norm) + 1./(2.*pow(R_norm,3));
		  C2 = 3./(4.*R_norm) - 3./(2.*pow(R_norm,3));
	  }
      else if (R_norm <= 2.0) {
		  C1 = 1. - (9./32.)*R_norm;
		  C2 = (3./32.)*R_norm;
	  }

      // Taken from Appendix C expression for M_UF.
      for (int l = 0; l < 3; ++l) {
        bp::numeric::array current_row =
            bp::extract<bp::numeric::array>(mobility[j*3 + l]);
        for (int m = 0; m < 3; ++m) {
	  // RPY PART
	  current_row[k*3 + m] +=(1.0/(6.0*pi*eta*a))*
	      ((l == m ? 1.0 : 0.0)*C1 + R[l]*R[m]/pow(R_norm,2)*C2);
	  // WALL PART
          current_row[k*3 + m] +=
              (1.0/(6.0*pi*eta*a))*
              (-0.25*(3.0*(1.0 - 6.0*h_hat*(1. - h_hat)*pow(e[2],2))/Rim_norm
                      - 6.0*(1.0 - 5.0*pow(e[2],2))/(pow(Rim_norm,3))
                      + 10.0*(1.0 - 7.0*pow(e[2],2))/(pow(Rim_norm,5)))*(e[l]*e[m])
               - (0.25*(3.0*(1.0 + 2.0*h_hat*(1. - h_hat)*pow(e[2],2))/Rim_norm
                        + 2.0*(1.0 - 3.0*pow(e[2],2))/(pow(Rim_norm,3))
                        - 2.0*(1.0 - 5.0*pow(e[2],2))/(pow(Rim_norm,5))))*(l == m ? 1.0 : 0.0)
               + 0.5*(3.0*h_hat*(1. - 6.0*(1. - h_hat)*pow(e[2],2))/Rim_norm
                      - 6.0*(1.0 - 5.0*pow(e[2],2))/(pow(Rim_norm,3))
                      + 10.0*(2.0 - 7.0*pow(e[2],2))/(pow(Rim_norm,5)))*(e[l]*e_3[m])
               + 0.5*(3.0*h_hat/Rim_norm - 10./(pow(Rim_norm,5)))*(e_3[l]*e[m])
               - (3.0*(pow(h_hat,2))*(pow(e[2],2))/Rim_norm 
                  + 3.0*(pow(e[2],2))/(pow(Rim_norm,3))
                  + (2.0 - 15.0*pow(e[2],2))/(pow(Rim_norm,5)))*
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
	current_row[j*3 + m] += (1./(6.0*pi*eta*a))*(l == m ? 1.0 : 0.0);
        current_row[j*3 + m] += (1./(6.0*pi*eta*a))*(
            (l == m ? 1.0 : 0.0)*(l != 2 ? 1.0 : 0.0)*(-1./16.)*
            (9./h - 2./(pow(h,3)) + 1./(pow(h,5)))
            + (l == m ? 1.0 : 0.0 )*(l == 2 ? 1.0 : 0.0)*(-1./8.)*
            (9./h - 4./(pow(h,3)) + 1./(pow(h,5))));
      }
    }
  }
}


void RPYInfiniteFluidMobility(bp::list r_vectors, 
                             double eta,
                             double a, int num_particles,
                             bp::numeric::array mobility) {
  // Create the mobility of particles in a fluid with a single wall at z = 0.
  double pi = 3.1415926535897932;
  double C1, C2;
  double* R;
  R = new double[3];
  for (int j = 0; j < num_particles; ++j) {
    bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(r_vectors[j]);
    for (int k = j+1; k < num_particles; ++k) {
      // Here notation is based on appendix C of the Swan and Brady paper:
      //  'Simulation of hydrodynamically interacting particles near a no-slip
      //   boundary.'
      bp::numeric::array  r_vector_2 = bp::extract<bp::numeric::array>(r_vectors[k]);
      for (int l = 0; l < 3; ++l) {
        R[l] = (bp::extract<double>(r_vector_1[l]) -
                bp::extract<double>(r_vector_2[l]))/a;
      }

              
      double R_norm = 0.0;
      for (int l = 0; l < 3; ++l) {
        R_norm += R[l]*R[l];
      }
      R_norm = sqrt(R_norm);


      if (R_norm > 2.0) {
		  C1 = 3./(4.*R_norm) + 1./(2.*pow(R_norm,3));
		  C2 = 3./(4.*R_norm) - 3./(2.*pow(R_norm,3));
	  }
      else if (R_norm <= 2.0) {
		  C1 = 1. - (9./32.)*R_norm;
		  C2 = (3./32.)*R_norm;
	  }

      // Taken from Appendix C expression for M_UF.
      for (int l = 0; l < 3; ++l) {
        bp::numeric::array current_row =
            bp::extract<bp::numeric::array>(mobility[j*3 + l]);
        for (int m = 0; m < 3; ++m) {
	  // RPY PART
	  current_row[k*3 + m] +=(1.0/(6.0*pi*eta*a))*
	      ((l == m ? 1.0 : 0.0)*C1 + R[l]*R[m]/pow(R_norm,2)*C2);	 
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
    for (int l = 0; l < 3; ++l) {
      bp::numeric::array current_row =
          bp::extract<bp::numeric::array>(mobility[j*3 + l]);
      for (int m = 0; m < 3; ++m) {
	current_row[j*3 + m] += (1./(6.0*pi*eta*a))*(l == m ? 1.0 : 0.0);
      }
    }
  }
}

void MobilityVectorProductOneParticle(bp::list r_vectors, 
			   double eta,
			   double a, int num_particles,
                           bp::numeric::array vector,
                           bp::numeric::array vector_res,
			   int index_particle) {
  // Create the mobility of particles in a fluid with a single wall at z = 0.

  double pi = 3.1415926535897932;
  double C1, C2;
  double Mlm;
  double* R;
  double* Rim;
  R = new double[3];
  Rim = new double[3];
  //double vectemp[num_particles*3];
  //for(int i =0; i< num_particles*3; ++i){
	//  vectemp[i]=0.0;
  //}
  double h = 0.0;
  int j = index_particle;
  
  bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(r_vectors[j]);

  for (int k = 0; k < num_particles; ++k){
    if(k == j){
      continue;
    }

    // Here notation is based on appendix C of the Swan and Brady paper:
    //  'Simulation of hydrodynamically interacting particles near a no-slip
    //   boundary.'
    bp::numeric::array  r_vector_2 = bp::extract<bp::numeric::array>(r_vectors[k]);
    h = bp::extract<double>(r_vector_2[2]);
    for (int l = 0; l < 3; ++l) {
      R[l] = (bp::extract<double>(r_vector_1[l]) -
	      bp::extract<double>(r_vector_2[l]))/a;
    }
    for (int l = 0; l < 2; ++l) {
      Rim[l] = (bp::extract<double>(r_vector_1[l]) -
	      bp::extract<double>(r_vector_2[l]))/a;
    }
    Rim[2] = (bp::extract<double>(r_vector_1[2]) -
	    bp::extract<double>(r_vector_2[2]) + 2.0*h)/a;
	    
    double R_norm = 0.0;
    double Rim_norm = 0.0;
    for (int l = 0; l < 3; ++l) {
      R_norm += R[l]*R[l];
      Rim_norm += Rim[l]*Rim[l];
    }
    R_norm = sqrt(R_norm);

    Rim_norm = sqrt(Rim_norm);
    double* e = new double[3];
    for (int l = 0; l < 3; ++l) {
      e[l] = Rim[l]/Rim_norm;
    }
    double* e_3 = new double[3];
    e_3[0] = 0.0;
    e_3[1] = 0.0;
    e_3[2] = e[2];
    
    double h_hat = h/(a*Rim[2]);
    
    if (R_norm > 2.0) {
		C1 = 3./(4.*R_norm) + 1./(2.*pow(R_norm,3));
		C2 = 3./(4.*R_norm) - 3./(2.*pow(R_norm,3));
	}
	else if (R_norm <= 2.0) {
		C1 = 1. - 9./32.*R_norm;
		C2 = 3./32.*R_norm;
	}

    for (int l = 0; l < 3; ++l) {
      for (int m = 0; m < 3; ++m) {	
    // Usual RPY Tensor
	Mlm = (l == m ? 1.0 : 0.0)*C1 + R[l]*R[m]/pow(R_norm,2)*C2; 
    // Taken from Appendix C expression for M_UF.
	Mlm +=
	    (-0.25*(3.0*(1.0 - 6.0*h_hat*(1. - h_hat)*pow(e[2],2))/Rim_norm
		    - 6.0*(1.0 - 5.0*pow(e[2],2))/(pow(Rim_norm,3))
		    + 10.0*(1.0 - 7.0*pow(e[2],2))/(pow(Rim_norm,5)))*(e[l]*e[m])
	      - (0.25*(3.0*(1.0 + 2.0*h_hat*(1. - h_hat)*pow(e[2],2))/Rim_norm
		      + 2.0*(1.0 - 3.0*pow(e[2],2))/(pow(Rim_norm,3))
		      - 2.0*(1.0 - 5.0*pow(e[2],2))/(pow(Rim_norm,5))))*(l == m ? 1.0 : 0.0)
	      + 0.5*(3.0*h_hat*(1. - 6.0*(1. - h_hat)*pow(e[2],2))/Rim_norm
		    - 6.0*(1.0 - 5.0*pow(e[2],2))/(pow(Rim_norm,3))
		    + 10.0*(2.0 - 7.0*pow(e[2],2))/(pow(Rim_norm,5)))*(e[l]*e_3[m])
	      + 0.5*(3.0*h_hat/Rim_norm - 10./(pow(Rim_norm,5)))*(e_3[l]*e[m])
	      - (3.0*(pow(h_hat,2))*(pow(e[2],2))/Rim_norm 
		+ 3.0*(pow(e[2],2))/(pow(Rim_norm,3))
		+ (2.0 - 15.0*pow(e[2],2))/(pow(Rim_norm,5)))*
	      (e_3[l]*e_3[m])/(pow(e[2],2)));
	      
	Mlm = Mlm*(1.0/(6.0*pi*eta*a));
	
	vector_res[l] += Mlm*bp::extract<double>(vector[k*3+m]);
      }
    }
  }
  
  
  // Diagonal blocks, self mobility.
  h = bp::extract<double>(r_vector_1[2])/a;
  for (int l = 0; l < 3; ++l) {
    for (int m = 0; m < 3; ++m) {
         // RPY
	  Mlm = (l == m ? 1.0 : 0.0);
	  // Wall
	  Mlm += (
	  (l == m ? 1.0 : 0.0)*(l != 2 ? 1.0 : 0.0)*(-1./16.)*
	  (9./h - 2./(pow(h,3)) + 1./(pow(h,5)))
	  + (l == m ? 1.0 : 0.0 )*(l == 2 ? 1.0 : 0.0)*(-1./8.)*
	  (9./h - 4./(pow(h,3)) + 1./(pow(h,5))));
	  Mlm = Mlm*(1./(6.0*pi*eta*a));
	  vector_res[l]+= Mlm*bp::extract<double>(vector[j*3+m]);
    }
  } 
}



BOOST_PYTHON_MODULE(mobility_ext)
{
  using namespace boost::python;
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  def("single_wall_fluid_mobility", SingleWallFluidMobilityCorrection);
  def("RPY_single_wall_fluid_mobility", RPYSingleWallFluidMobility);
  def("RPY_infinite_fluid_mobility", RPYInfiniteFluidMobility);
  def("mobility_vector_product", MobilityVectorProduct);
  def("mobility_vector_product_one_particle", MobilityVectorProductOneParticle);

}

