// OLD CODE; DO NOT USE!!!
// USE INSTEAD THE CODE ON mobility.cpp
// 
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
  double R[3];
  double Rim[3];
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
      double e[3];
      for (int l = 0; l < 3; ++l) {
        e[l] = Rim[l]/Rim_norm;
      }
      double e_3[3];
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
void MobilityVectorProduct(bp::numeric::array r_vectors, 
                           double eta,
                           double a, 
                           int num_particles,
                           bp::numeric::array periodic_length,
                           bp::numeric::array vector,
                           bp::numeric::array vector_res ) {
  // Create the mobility of particles in a fluid with a single wall at z = 0.
  double pi = 3.1415926535897932;
  double C1, C2;
  double Mlm;
  double R[3];
  double Rim[3];
  double h = 0.0;

  // Determine if the space is pseudo-periodic in the directions x or y
  // We use a extended unit cell of length L=3*(Lx, Ly)
  bp::numeric::array L = bp::extract<bp::numeric::array>(periodic_length);  
  int periodic[2] = {0,0};
  int box[2];
  for(int l=0; l<2; l++){
    if(bp::extract<double>(L[l]) > 0){
      periodic[l] = 1;
    }
  }

  for (int j = 0; j < num_particles; ++j) {
    bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(r_vectors[j]);
    // Loop over image boxes and then over particles
    for(box[0] = -periodic[0]; box[0] <= periodic[0]; box[0]++){
      for(box[1] = -periodic[1]; box[1] <= periodic[1]; box[1]++){
        for (int k = j; k < num_particles; ++k) {
          // Here notation is based on appendix C of the Swan and Brady paper:
          //  'Simulation of hydrodynamically interacting particles near a no-slip
          //   boundary.'
          bp::numeric::array  r_vector_2 = bp::extract<bp::numeric::array>(r_vectors[k]);
          h = bp::extract<double>(r_vector_2[2]);
          for(int l = 0; l < 3; ++l){
            R[l] = (bp::extract<double>(r_vector_1[l]) - bp::extract<double>(r_vector_2[l]));
            Rim[l] = R[l];
          }
          Rim[2] = (bp::extract<double>(r_vector_1[2]) - bp::extract<double>(r_vector_2[2]) + 2.0*h);
	  
          // Project a vector r to the extended unit cell
          // centered around (0,0) and of size L=3*(Lx, Ly). If 
          // any dimension of L is equal or smaller than zero the 
          // box is assumed to be infinite in that direction.
          for(int l=0; l<2; ++l){
            if(bp::extract<double>(L[l]) > 0){
              R[l] = R[l] - int(R[l] / bp::extract<double>(L[l]) + 0.5 * (int(R[l]>0) - int(R[l]<0))) * bp::extract<double>(L[l]);
              R[l] = R[l] + box[l] * bp::extract<double>(L[l]);
              Rim[l] = R[l];
            }
          }
	  
          // Scale distances
          for(int l=0; l<3; l++){
            R[l] = R[l] / a;
            Rim[l] = Rim[l] / a;
          }

          double R_norm = 0.0;
          double Rim_norm = 0.0;
          for (int l = 0; l < 3; ++l) {
            R_norm += R[l]*R[l];
            Rim_norm += Rim[l]*Rim[l];
          }
          R_norm = sqrt(R_norm);
	  
          Rim_norm = sqrt(Rim_norm);
          double e[3];
          for (int l = 0; l < 3; ++l) {
            e[l] = Rim[l]/Rim_norm;
          }
          double e_3[3];
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
	      
              Mlm = Mlm / (6.0*pi*eta*a);
              if((j != k) or (box[0] != 0) or (box[1] !=0)){
                vector_res[j*3+l] += Mlm*bp::extract<double>(vector[k*3+m]);
                // Use the fact that M is symmetric 
                if(j != k){
                  vector_res[k*3+m] += Mlm*bp::extract<double>(vector[j*3+l]); 
                }
              }
            }
          }
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


void NoWallMobilityVectorProduct(bp::numeric::array r_vectors, 
                                 double eta,
                                 double a, 
                                 int num_particles,
                                 bp::numeric::array periodic_length,
                                 bp::numeric::array vector,
                                 bp::numeric::array vector_res ){
  double pi = 3.1415926535897932;
  double C1, C2;
  double Mlm;
  double R[3];
  double Rim[3];
  double h = 0.0;

  // Determine if the space is pseudo-periodic in the directions x or y
  // We use a extended unit cell of length L=3*(Lx, Ly)
  bp::numeric::array L = bp::extract<bp::numeric::array>(periodic_length);  
  int periodic[2] = {0,0};
  int box[2];
  for(int l=0; l<2; l++){
    if(bp::extract<double>(L[l]) > 0){
      periodic[l] = 1;
    }
  }

  for (int j = 0; j < num_particles; ++j) {
    bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(r_vectors[j]);
    // Loop over image boxes and then over particles
    for(box[0] = -periodic[0]; box[0] <= periodic[0]; box[0]++){
      for(box[1] = -periodic[1]; box[1] <= periodic[1]; box[1]++){
        for (int k = j; k < num_particles; ++k) {
          bp::numeric::array  r_vector_2 = bp::extract<bp::numeric::array>(r_vectors[k]);
          for(int l = 0; l < 3; ++l){
            R[l] = (bp::extract<double>(r_vector_1[l]) - bp::extract<double>(r_vector_2[l]));
          }
	  
          // Project a vector r to the extended unit cell
          // centered around (0,0) and of size L=3*(Lx, Ly). If 
          // any dimension of L is equal or smaller than zero the 
          // box is assumed to be infinite in that direction.
          for(int l=0; l<2; ++l){
            if(bp::extract<double>(L[l]) > 0){
              R[l] = R[l] - int(R[l] / bp::extract<double>(L[l]) + 0.5 * (int(R[l]>0) - int(R[l]<0))) * bp::extract<double>(L[l]);
              R[l] = R[l] + box[l] * bp::extract<double>(L[l]);
            }
          }
	  
          // Scale distances
          for(int l=0; l<3; l++){
            R[l] = R[l] / a;
            Rim[l] = Rim[l] / a;
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
            C1 = 1. - 9./32.*R_norm;
            C2 = 3./32.*R_norm;
          }
	  
          for (int l = 0; l < 3; ++l) {
            for (int m = 0; m < 3; ++m) {	
              // Usual RPY Tensor
              Mlm = (l == m ? 1.0 : 0.0)*C1 + R[l]*R[m]/pow(R_norm,2)*C2;       
              Mlm = Mlm / (6.0*pi*eta*a);
              if((j != k) or (box[0] != 0) or (box[1] !=0)){
                vector_res[j*3+l] += Mlm*bp::extract<double>(vector[k*3+m]);
                // Use the fact that M is symmetric 
                if(j != k){
                  vector_res[k*3+m] += Mlm*bp::extract<double>(vector[j*3+l]); 
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Diagonal blocks, self mobility.
  for (int j = 0; j < num_particles; ++j) {
    bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(r_vectors[j]);
    h = bp::extract<double>(r_vector_1[2])/a;
    for (int l = 0; l < 3; ++l) {
      Mlm = 1.0 / (6.0 * pi * eta * a);
      vector_res[j*3+l]+= Mlm*bp::extract<double>(vector[j*3+l]);
    }
  }
}


void SingleWallFluidMobilityCorrection(bp::list r_vectors, 
                                       double eta,
                                       double a, int num_particles,
                                       bp::numeric::array mobility) {
  double pi = 3.1415926535897932;
  // Create the mobility of particles in a fluid with a single wall at z = 0.
  double R[3];
  
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
      double e[3];
      for (int l = 0; l < 3; ++l) {
        e[l] = R[l]/R_norm;
      }
      double e_3[3];
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
  double R[3];
  double Rim[3];
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
      double e[3];
      for (int l = 0; l < 3; ++l) {
        e[l] = Rim[l]/Rim_norm;
      }
      double e_3[3];
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


void RPYInfiniteFluidMobility(bp::numeric::array r_vectors,
                              double eta,
                              double a, int num_particles,
                              bp::numeric::array mobility) {
  // Create the mobility of particles in a fluid with a single wall at z = 0.
  double pi = 3.1415926535897932;
  double C1, C2;
  double R[3];
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
  double R[3];
  double Rim[3];
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
    double e[3];
    for (int l = 0; l < 3; ++l) {
      e[l] = Rim[l]/Rim_norm;
    }
    double e_3[3];
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


void MobilityVectorProductSourceTargetOneWall(bp::numeric::array source, 
                                              bp::numeric::array target,
                                              bp::numeric::array force,
                                              bp::numeric::array radius_source,
                                              bp::numeric::array radius_target,
                                              bp::numeric::array velocity,
                                              bp::numeric::array periodic_length,
                                              double eta,
                                              int num_sources,
                                              int num_targets){

  // Create the mobility of particles in a fluid with a single wall at z = 0.
  double pi = 3.1415926535897932;
  double C1, C2;
  double Mlm;
  double R[3], RR[3][3];
  double Rim[3], RRim[3][3];
  double h = 0.0;
  double delta_3[3] = {0, 0, 1};
  double prefactor = 1.0 / (8 * pi * eta);
  bp::numeric::array radius_target_ext = bp::extract<bp::numeric::array>(radius_target);
  bp::numeric::array radius_source_ext = bp::extract<bp::numeric::array>(radius_source);
  
  double I[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
  double J[3][3] = {{0,0,0},{0,0,0},{0,0,1}};
  double P[3][3] = {{1,0,0},{0,1,0},{0,0,-1}};

  // Determine if the space is pseudo-periodic in the directions x or y
  // We use a extended unit cell of length L=3*(Lx, Ly)
  bp::numeric::array L = bp::extract<bp::numeric::array>(periodic_length);  
  int periodic[2] = {0,0};
  int box[2];
  for(int l=0; l<2; l++){
    if(bp::extract<double>(L[l]) > 0){
      periodic[l] = 1;
    }
  }

  // Loop over targets
  for (int j = 0; j < num_targets; ++j) {
    bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(target[j]);

    // Loop over image boxes and then over sources
    for(box[0] = -periodic[0]; box[0] <= periodic[0]; box[0]++){
      for(box[1] = -periodic[1]; box[1] <= periodic[1]; box[1]++){
        for (int k = 0; k < num_sources; ++k) {
          // Here notation is based on appendix C of the Swan and Brady paper
          // but generalized to particles with different radius:
          // 'Simulation of hydrodynamically interacting particles near a no-slip
          //  boundary.'
          bp::numeric::array  r_vector_2 = bp::extract<bp::numeric::array>(source[k]);
          h = bp::extract<double>(r_vector_2[2]);
          for(int l = 0; l < 3; ++l){
            R[l] = (bp::extract<double>(r_vector_1[l]) - bp::extract<double>(r_vector_2[l]));
            Rim[l] = R[l];
          }
          Rim[2] = (bp::extract<double>(r_vector_1[2]) - bp::extract<double>(r_vector_2[2]) + 2.0*h);
	  
          // Project a vector r to the extended unit cell
          // centered around (0,0) and of size L=3*(Lx, Ly). If 
          // any dimension of L is equal or smaller than zero the 
          // box is assumed to be infinite in that direction.
          for(int l=0; l<2; ++l){
            if(bp::extract<double>(L[l]) > 0){
              R[l] = R[l] - int(R[l] / bp::extract<double>(L[l]) + 0.5 * (int(R[l]>0) - int(R[l]<0))) * bp::extract<double>(L[l]);
              R[l] = R[l] + box[l] * bp::extract<double>(L[l]);
              Rim[l] = R[l];
            }
          }
	  
          // Compute distances
          double x3[3] = {0,0, bp::extract<double>(r_vector_1[2])};
          double y3[3] = {0,0, bp::extract<double>(r_vector_2[2])};
          double vec_Rim3[3] = {0,0,Rim[2]};
          double R2 = 0.0;
          double Rim2 = 0.0;
          for (int l = 0; l < 3; ++l) {
            R2 += R[l]*R[l];
            Rim2 += Rim[l]*Rim[l];
          }
          double R_norm = sqrt(R2);  
          double Rim_norm = sqrt(Rim2);
          double Rim3 = Rim2 * Rim_norm;
          double Rim5 = Rim2 * Rim3;
          double Rim7 = Rim2 * Rim5;
          double Rim9 = Rim2 * Rim7;

          // Form tensors
          for(int l=0; l<3; l++){
            for(int m=0; m<3; m++){
              RR[l][m] = R[l] * R[m];
              RRim[l][m] = Rim[l] * Rim[m];
            }
          }

          // Compute unbounded contribution constants
          double b = bp::extract<double>(radius_target_ext[j]);
          double a = bp::extract<double>(radius_source_ext[k]);
          double b2 = b * b;
          double a2 = a * a;
          if(R_norm > (b+a)){
            C1 = (1 + (b2+a2) / (3 * R2)) * (prefactor / R_norm);
            C2 = ((1 - (b2+a2) / R2) / R2) * (prefactor / R_norm);
          }
          else if(R_norm > fabs(b-a)){
            double R3 = R2 * R_norm;
            C1 = ((16*(b+a)*R3 - pow(pow(b-a,2) + 3*R2,2)) / (32*R3)) / (6 * pi * eta * b * a);
            C2 = ((3*pow(pow(b-a,2)-R2, 2) / (32*R3)) / R2) / (6 * pi * eta * b * a); 

          }
          else{
            double largest_radius = (a > b) ? a : b;    
            C1 = 1.0 / (6 * pi * eta * largest_radius);
            C2 = 0;
          }

          // Build tensor
          double M[3][3] = {0};
          for (int l = 0; l < 3; ++l) {
            for (int m = 0; m < 3; ++m) {	      
              // Wall contribution
              M[l][m] += ((1+(b2+a2)/(3.0*Rim2)) * I[l][m] + (1-(b2+a2)/Rim2) * RRim[l][m] / Rim2) / Rim_norm;
              M[l][m] += 2*(-J[l][m]/Rim_norm - Rim[l]*x3[m]/Rim3 - y3[l]*Rim[m]/Rim3 + x3[2]*y3[2]*(I[l][m]/Rim3 - 3*Rim[l]*Rim[m]/Rim5));
              M[l][m] += (2*b2/3.0) * (-J[l][m]/Rim3 + 3*Rim[l]*vec_Rim3[m]/Rim5 - y3[2]*(3*vec_Rim3[2]*I[l][m]/Rim5 + 3*delta_3[l]*Rim[m]/Rim5 + 3*Rim[l]*delta_3[m]/Rim5 - 15*vec_Rim3[2]*Rim[l]*Rim[m]/Rim7));
              M[l][m] += (2*a2/3.0) * (-J[l][m]/Rim3 + 3*vec_Rim3[l]*Rim[m]/Rim5 - x3[2]*(3*vec_Rim3[2]*I[l][m]/Rim5 + 3*delta_3[l]*Rim[m]/Rim5 + 3*Rim[l]*delta_3[m]/Rim5 - 15*vec_Rim3[2]*Rim[l]*Rim[m]/Rim7));
              M[l][m] += (2*b2*a2/3.0) * (-I[l][m]/Rim5 + 5*vec_Rim3[2]*vec_Rim3[2]*I[l][m]/Rim7 - J[l][m]/Rim5 + 5*vec_Rim3[l]*Rim[m]/Rim7 - J[l][m]/Rim5 + 5*Rim[l]*vec_Rim3[m]/Rim7 + 5*vec_Rim3[l]*Rim[m]/Rim7 + 5*Rim[l]*Rim[m]/Rim7 + 5*Rim[l]*vec_Rim3[m]/Rim7 -35 * vec_Rim3[2]*vec_Rim3[2]*Rim[l]*Rim[m]/Rim9);      
            }
          }
          // Multiply by P
          for(int l=0;l<3;l++){
            M[l][2] = -M[l][2];
          }
          // Multiply by prefactor and add unbounded contribution
          for(int l=0;l<3;l++){
            for(int m=0; m<3;m++){
              M[l][m] *= -prefactor;
              // Usual RPY Tensor (unbounded contribution)
              M[l][m] += (l == m ? 1.0 : 0.0)*C1 + R[l]*R[m]*C2; 
            }
          }
          // Compute velocity
          for(int l=0;l<3;l++){
            for(int m=0;m<3;m++){
              velocity[j*3+l] += M[l][m]*bp::extract<double>(force[k*3+m]);
            }
          }
        }
      }
    }
  }
}

     
void RPYFreeSurfaceCorrectionMobility(bp::numeric::array r_vectors,
                                      double eta,
                                      double a, int num_particles,
                                      bp::numeric::array mobility) {
  // Create the mobility of particles in a fluid with a single wall at z = 0.
  double pi = 3.1415926535897932;
  double C1, C2;
  double R[3], Rim[3];
  for (int j = 0; j < num_particles; ++j) {
    bp::numeric::array r_vector_1 = bp::extract<bp::numeric::array>(r_vectors[j]);
    for (int k = j; k < num_particles; ++k) {
      // Here notation is based on appendix C of the Swan and Brady paper:
      //  'Simulation of hydrodynamically interacting particles near a no-slip
      //   boundary.'
      bp::numeric::array  r_vector_2 = bp::extract<bp::numeric::array>(r_vectors[k]);
      for (int l = 0; l < 2; ++l) {
        R[l] = (bp::extract<double>(r_vector_1[l]) -
                bp::extract<double>(r_vector_2[l]))/a;
        
      }
      R[2] = (bp::extract<double>(r_vector_1[2]) +
              bp::extract<double>(r_vector_2[2]))/a;

              
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
}







BOOST_PYTHON_MODULE(mobility_ext)
{
  using namespace boost::python;
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  def("single_wall_fluid_mobility", SingleWallFluidMobilityCorrection);
  def("RPY_single_wall_fluid_mobility", RPYSingleWallFluidMobility);
  def("RPY_infinite_fluid_mobility", RPYInfiniteFluidMobility);
  def("mobility_vector_product", MobilityVectorProduct);
  def("no_wall_mobility_vector_product", NoWallMobilityVectorProduct);
  def("mobility_vector_product_one_particle", MobilityVectorProductOneParticle);
  def("mobility_vector_product_source_target_one_wall", MobilityVectorProductSourceTargetOneWall);
  def("RPY_free_surface_correction_mobility", RPYFreeSurfaceCorrectionMobility);
}

// PYTHON INTERFACES TO THE ABOVE FUNCTIONS

// def boosted_single_wall_fluid_mobility(r_vectors, eta, a, *args, **kwargs):
//   '''
//   Same as single wall fluid mobility, but boosted into C++ for
//   a speedup. Must compile mobility_ext.cc before this will work
//   (use Makefile).

//   For blobs overlaping the wall we use
//   Compute M = B^T * M_tilde(z_effective) * B.
//   '''
//   # Set effective height
//   r_vectors_effective = shift_heights(r_vectors, a)

//   # Compute damping matrix B
//   B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)

//   num_particles = r_vectors.size // 3
//   fluid_mobility = np.zeros( (num_particles*3, num_particles*3) )
//   me.RPY_single_wall_fluid_mobility(np.reshape(r_vectors_effective, (num_particles, 3)), eta, a, num_particles, fluid_mobility)

//   # Compute M = B^T * M_tilde * B
//   if overlap is True:
//     return B.dot( (B.dot(fluid_mobility.T)).T )
//   else:
//     return fluid_mobility

// def boosted_infinite_fluid_mobility(r_vectors, eta, a, *args, **kwargs):
//   '''
//   Same as rotne_prager_tensor, but boosted into C++ for
//   a speedup. Must compile mobility_ext.cc before this will work
//   (use Makefile).
//   '''
//   num_particles = len(r_vectors)
//   # fluid_mobility = np.array([np.zeros(3*num_particles) for _ in range(3*num_particles)])
//   fluid_mobility = np.zeros((num_particles*3, num_particles*3))
//   me.RPY_infinite_fluid_mobility(r_vectors, eta, a, num_particles, fluid_mobility)
//   return fluid_mobility


// def boosted_mobility_vector_product(r_vectors, vector, eta, a, *args, **kwargs):
//   '''
//   Compute a mobility * vector product boosted in C++ for a
//   speedup. It includes wall corrections.
//   Must compile mobility_ext.cc before this will work
//   (use Makefile).

//   For blobs overlaping the wall we use
//   Compute M = B^T * M_tilde(z_effective) * B.
//   '''
//   ## THE USE OF VECTOR_RES AS THE RESULT OF THE MATRIX VECTOR PRODUCT IS
//   ## TEMPORARY: I NEED TO FIGURE OUT HOW TO CONVERT A DOUBLE TO A NUMPY ARRAY
//   ## WITH BOOST
//   L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
//   # Get effective height
//   r_vectors_effective = shift_heights(r_vectors, a)
//   # Compute damping matrix B
//   B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
//   # Compute B * vector
//   if overlap is True:
//     vector = B.dot(vector)
//   # Compute M_tilde * B * vector
//   num_particles = r_vectors.size // 3
//   vector_res = np.zeros(r_vectors.size)
//   r_vec_for_mob = np.reshape(r_vectors_effective, (r_vectors_effective.size // 3, 3))
//   me.mobility_vector_product(r_vec_for_mob, eta, a, num_particles, L, vector, vector_res)
//   # Compute B.T * M * B * vector
//   if overlap is True:
//     vector_res = B.dot(vector_res)
//   return vector_res


// def boosted_no_wall_mobility_vector_product(r_vectors, vector, eta, a, *args, **kwargs):
//   '''
//   Compute a mobility * vector product boosted in C++ for a
//   speedup. It uses the RPY tensor.
//   Must compile mobility_ext.cc before this will work
//   (use Makefile).
//   '''
//   L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
//   num_particles = r_vectors.size // 3
//   vector_res = np.zeros(r_vectors.size)
//   r_vec_for_mob = np.reshape(r_vectors, (r_vectors.size // 3, 3))
//   me.no_wall_mobility_vector_product(r_vec_for_mob, eta, a, num_particles, L, vector, vector_res)
//   return vector_res


// def boosted_mobility_vector_product_one_particle(r_vectors, eta, a, vector, index_particle, *args, **kwargs):
//   '''
//   Compute a mobility * vector product for only one particle. Return the
//   velocity of of the desired particle. It includes wall corrections.
//   Boosted in C++ for a speedup. Must compile mobility_ext.cc before this
//   will work (use Makefile).
//   '''
//   num_particles = len(r_vectors)
//   ## THE USE OF VECTOR_RES AS THE RESULT OF THE MATRIX VECTOR PRODUCT IS
//   ## TEMPORARY: I NEED TO FIGURE OUT HOW TO CONVERT A DOUBLE TO A NUMPY ARRAY
//   ## WITH BOOST
//   vector_res = np.zeros(3)
//   me.mobility_vector_product_one_particle(r_vectors, eta, a, \
// 					  num_particles, vector, \
// 					  vector_res, index_particle)
//   return vector_res


// def boosted_mobility_vector_product_source_target(source, target, force, radius_source, radius_target, eta, *args, **kwargs):
//   '''
//   Compute a mobility * vector product boosted in C++ for a
//   speedup. It includes wall corrections.
//   Must compile mobility_ext.cc before this will work
//   (use Makefile).

//   For blobs overlaping the wall we use
//   Compute M = B^T * M_tilde(z_effective) * B.
//   '''
//   L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

//   # Compute effective heights
//   x = shift_heights_different_radius(target, radius_target)
//   y = shift_heights_different_radius(source, radius_source)

//   # Compute dumping matrices
//   B_target, overlap_target = damping_matrix_B_different_radius(target, radius_target, *args, **kwargs)
//   B_source, overlap_source = damping_matrix_B_different_radius(source, radius_source, *args, **kwargs)

//   # Compute B * vector
//   if overlap_source is True:
//     force = B_source.dot(force.flatten())

//   # Compute M_tilde * B * vector
//   num_sources = source.size // 3
//   num_targets = target.size // 3
//   vector_res = np.zeros(target.size)
//   x_for_mob = np.reshape(x, (x.size // 3, 3))
//   y_for_mob = np.reshape(y, (y.size // 3, 3))
//   force = np.reshape(force, force.size)

//   me.mobility_vector_product_source_target_one_wall(y_for_mob, x_for_mob, force, radius_source, radius_target, vector_res, L, eta, num_sources, num_targets)

//   # Compute B.T * M * B * vector
//   if overlap_target is True:
//     vector_res = B_target.dot(vector_res)
//   return vector_res



