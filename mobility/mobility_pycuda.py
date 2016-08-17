import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
#include <stdio.h>
/*
 mobilityUFRPY computes the 3x3 RPY mobility
 between blobs i and j normalized with 8 pi eta a
*/
__device__ void mobilityUFRPY(double rx,
			      double ry,
			      double rz,
			      double &Mxx,
			      double &Mxy,
			      double &Mxz,
			      double &Myy,
			      double &Myz,
			      double &Mzz,
			      int i,
                              int j,
                              double invaGPU){
  
  double fourOverThree = 4.0 / 3.0;

  if(i == j){
    Mxx = fourOverThree;
    Mxy = 0;
    Mxz = 0;
    Myy = Mxx;
    Myz = 0;
    Mzz = Mxx;
  }
  else{
    rx = rx * invaGPU; //Normalize distance with hydrodynamic radius
    ry = ry * invaGPU;
    rz = rz * invaGPU;
    double r2 = rx*rx + ry*ry + rz*rz;
    double r = sqrt(r2);
    //We should not divide by zero but std::numeric_limits<double>::min() does not work in the GPU
    //double invr = (r > std::numeric_limits<double>::min()) ? (1.0 / r) : (1.0 / std::numeric_limits<double>::min())
    double invr = 1.0 / r;
    double invr2 = invr * invr;
    double c1, c2;
    if(r>=2){
      c1 = 1 + 2 / (3 * r2);
      c2 = (1 - 2 * invr2) * invr2;
      Mxx = (c1 + c2*rx*rx) * invr;
      Mxy = (     c2*rx*ry) * invr;
      Mxz = (     c2*rx*rz) * invr;
      Myy = (c1 + c2*ry*ry) * invr;
      Myz = (     c2*ry*rz) * invr;
      Mzz = (c1 + c2*rz*rz) * invr;
    }
    else{
      c1 = fourOverThree * (1 - 0.28125 * r); // 9/32 = 0.28125
      c2 = fourOverThree * 0.09375 * invr;    // 3/32 = 0.09375
      Mxx = c1 + c2 * rx*rx ;
      Mxy =      c2 * rx*ry ;
      Mxz =      c2 * rx*rz ;
      Myy = c1 + c2 * ry*ry ;
      Myz =      c2 * ry*rz ;
      Mzz = c1 + c2 * rz*rz ;
    }
  } 
  return;
}


/*
 mobilityUFSingleWallCorrection computes the 3x3 mobility correction due to a wall
 between blobs i and j normalized with 8 pi eta a.
 This uses the expression from the Swan and Brady paper for a finite size particle.
 Mobility is normalize by 8*pi*eta*a.
*/
__device__ void mobilityUFSingleWallCorrection(double rx,
			                       double ry,
			                       double rz,
			                       double &Mxx,
                  			       double &Mxy,
			                       double &Mxz,
                                               double &Myx,
			                       double &Myy,
			                       double &Myz,
                                               double &Mzx,
                                               double &Mzy,
			                       double &Mzz,
			                       int i,
                                               int j,
                                               double invaGPU,
                                               double hj){

  if(i == j){
    double invZi = 1.0 / hj;
    Mxx += -(9*invZi - 2*pow(invZi,3) + pow(invZi,5)) / 12.0;
    Myy += -(9*invZi - 2*pow(invZi,3) + pow(invZi,5)) / 12.0;
    Mzz += -(9*invZi - 4*pow(invZi,3) + pow(invZi,5)) / 6.0;
  }
  else{
    double h_hat = hj / rz;
    double invR = rsqrt(rx*rx + ry*ry + rz*rz); // = 1 / r;
    double ex = rx * invR;
    double ey = ry * invR;
    double ez = rz * invR;
    
    double fact1 = -(3*(1+2*h_hat*(1-h_hat)*ez*ez) * invR + 2*(1-3*ez*ez) * pow(invR,3) - 2*(1-5*ez*ez) * pow(invR,5))  / 3.0;
    double fact2 = -(3*(1-6*h_hat*(1-h_hat)*ez*ez) * invR - 6*(1-5*ez*ez) * pow(invR,3) + 10*(1-7*ez*ez) * pow(invR,5)) / 3.0;
    double fact3 =  ez * (3*h_hat*(1-6*(1-h_hat)*ez*ez) * invR - 6*(1-5*ez*ez) * pow(invR,3) + 10*(2-7*ez*ez) * pow(invR,5)) * 2.0 / 3.0;
    double fact4 =  ez * (3*h_hat*invR - 10*pow(invR,5)) * 2.0 / 3.0;
    double fact5 = -(3*h_hat*h_hat*ez*ez*invR + 3*ez*ez*pow(invR, 3) + (2-15*ez*ez)*pow(invR, 5)) * 4.0 / 3.0;
    
    Mxx += fact1 + fact2 * ex*ex;
    Mxy += fact2 * ex*ey;
    Mxz += fact2 * ex*ez + fact3 * ex;
    Myx += fact2 * ey*ex;
    Myy += fact1 + fact2 * ey*ey;
    Myz += fact2 * ey*ez + fact3 * ey;
    Mzx += fact2 * ez*ex + fact4 * ex;
    Mzy += fact2 * ez*ey + fact4 * ey;
    Mzz += fact1 + fact2 * ez*ez + fact3 * ez + fact4 * ez + fact5;         
  }
}



/*
 velocity_from_force computes the product
 U = M*F
*/
__global__ void velocity_from_force(const double *x,
                                    const double *f,					
                                    double *u,
				    int number_of_blobs,
                                    double eta,
                                    double a,
                                    double Lx,
                                    double Ly,
                                    double Lz){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;   

  double invaGPU = 1.0 / a;

  double Ux=0;
  double Uy=0;
  double Uz=0;

  double rx, ry, rz;

  double Mxx, Mxy, Mxz;
  double Myx, Myy, Myz;
  double Mzx, Mzy, Mzz;

  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;

  // Determine if the space is pseudo-periodic in any dimension
  // We use a extended unit cell of length L=3*(Lx, Ly, Lz)
  int periodic_x = 0, periodic_y = 0, periodic_z = 0;
  if(Lx > 0){
    periodic_x = 1;
  }
  if(Ly > 0){
    periodic_y = 1;
  }
  if(Lz > 0){
    periodic_z = 1;
  }
  
  // Loop over image boxes and then over particles
  for(int boxX = -periodic_x; boxX <= periodic_x; boxX++){
    for(int boxY = -periodic_y; boxY <= periodic_y; boxY++){
      for(int boxZ = -periodic_z; boxZ <= periodic_z; boxZ++){
	for(int j=0; j<number_of_blobs; j++){
	  joffset = j * NDIM;
	  
	  // Compute vector between particles i and j
	  // rx = x[ioffset    ] - (x[joffset    ] + boxX * Lx);
	  // ry = x[ioffset + 1] - (x[joffset + 1] + boxY * Ly);
	  // rz = x[ioffset + 2] - (x[joffset + 2] + boxZ * Lz);
	  rx = x[ioffset    ] - x[joffset    ];
	  ry = x[ioffset + 1] - x[joffset + 1];
	  rz = x[ioffset + 2] - x[joffset + 2];

	  // Project a vector r to the extended unit cell
	  // centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If 
	  // any dimension of L is equal or smaller than zero the 
	  // box is assumed to be infinite in that direction.
	  if(Lx > 0){
	    rx = rx - int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + 0.5 * (int(rz>0) - int(rz<0))) * Lz;
            rz = rz + boxZ * Lz;
	  }
  
	  // 1. Compute mobility for pair i-j, if i==j use self-interation
          int j_image = j;
          if(boxX!=0 or boxY!=0 or boxZ!=0){
            j_image = -1;
          }
	  mobilityUFRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
	  Myx = Mxy;
	  Mzx = Mxz;
	  Mzy = Myz;
	  mobilityUFSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy,Mzz, i,j_image, invaGPU, x[joffset+2]/a);
	  
	  //2. Compute product M_ij * F_j
	  Ux = Ux + (Mxx * f[joffset] + Mxy * f[joffset + 1] + Mxz * f[joffset + 2]);
	  Uy = Uy + (Myx * f[joffset] + Myy * f[joffset + 1] + Myz * f[joffset + 2]);
	  Uz = Uz + (Mzx * f[joffset] + Mzy * f[joffset + 1] + Mzz * f[joffset + 2]);
	}
      }
    }
  }
  //LOOP END

  //3. Save velocity U_i
  double pi = 4.0 * atan(1.0);
  double norm_fact_f = 1.0 / (8 * pi * eta * a);
  u[ioffset    ] = Ux * norm_fact_f;
  u[ioffset + 1] = Uy * norm_fact_f;
  u[ioffset + 2] = Uz * norm_fact_f;

  return;
}



////////// WT //////////////////////////////////////////////////

/*
 mobilityWTRPY computes the 3x3 RPY mobility
 between blobs i and j normalized with 8 pi eta a**3
*/
__device__ void mobilityWTRPY(double rx,
			      double ry,
			      double rz,
			      double &Mxx,
			      double &Mxy,
			      double &Mxz,
			      double &Myy,
			      double &Myz,
			      double &Mzz,
			      int i,
			      int j,
                              double invaGPU){
  
  if(i==j){
    Mxx = 1.0;
    Mxy = 0;
    Mxz = 0;
    Myy = Mxx;
    Myz = 0;
    Mzz = Mxx;
  }
  else{
    rx = rx * invaGPU; //Normalize distance with hydrodynamic radius
    ry = ry * invaGPU;
    rz = rz * invaGPU;
    double r2 = rx*rx + ry*ry + rz*rz;
    double r = sqrt(r2);
    double r3 = r2*r;
    //We should not divide by zero but std::numeric_limits<double>::min() does not work in the GPU
    //double invr = (r > std::numeric_limits<double>::min()) ? (1.0 / r) : (1.0 / std::numeric_limits<double>::min())
    double invr = 1 / r;
    double invr2 = 1 / r2;
    double invr3 = 1 / r3;
    double c1, c2;
    if(r>=2){
      c1 = -0.5;
      c2 = 1.5 * invr2 ;
      Mxx = (c1 + c2*rx*rx) * invr3;
      Mxy = (     c2*rx*ry) * invr3;
      Mxz = (     c2*rx*rz) * invr3;
      Myy = (c1 + c2*ry*ry) * invr3;
      Myz = (     c2*ry*rz) * invr3;
      Mzz = (c1 + c2*rz*rz) * invr3;
    }
    else{
      c1 =  (1 - 0.84375 * r + 0.078125 * r3); // 27/32 = 0.84375, 5/64 = 0.078125
      c2 =  0.28125 * invr - 0.046875 * r;    // 9/32 = 0.28125, 3/64 = 0.046875
      Mxx = c1 + c2 * rx*rx ;
      Mxy =      c2 * rx*ry ;
      Mxz =      c2 * rx*rz ;
      Myy = c1 + c2 * ry*ry ;
      Myz =      c2 * ry*rz ;
      Mzz = c1 + c2 * rz*rz ;
    }
  } 
  return;
}

/*
 mobilityWTSingleWallCorrection computes the 3x3 mobility correction due to a wall
 between blobs i and j normalized with 8 pi eta a. 
 It maps torques to angular velocities.
 This uses the expression from the Swan and Brady paper for a finite size particle.
 Mobility is normalize by 8*pi*eta*a.
*/
__device__ void mobilityWTSingleWallCorrection(double rx,
			                       double ry,
			                       double rz,
			                       double &Mxx,
                  			       double &Mxy,
			                       double &Mxz,
                                               double &Myx,
			                       double &Myy,
			                       double &Myz,
                                               double &Mzx,
                                               double &Mzy,
			                       double &Mzz,
			                       int i,
			                       int j,
                                               double invaGPU,
                                               double hj){
  if(i == j){
    double invZi = 1.0 / hj;
    double invZi3 = pow(invZi,3);
    Mxx += - invZi3 * 0.3125; // 15/48 = 0.3125
    Myy += - invZi3 * 0.3125; // 15/48 = 0.3125
    Mzz += - invZi3 * 0.125; // 3/24 = 0.125
  }
  else{
    double invR = rsqrt(rx*rx + ry*ry + rz*rz); // = 1 / r;
    double invR3 = invR * invR * invR;
    double ex = rx * invR;
    double ey = ry * invR;
    double ez = rz * invR;
    
    double fact1 =  ((1-6*ez*ez) * invR3 ) / 2.0;
    double fact2 = -(9 * invR3) / 6.0;
    double fact3 =  (3 * invR3 * ez);
    double fact4 =  (3 * invR3);
    
    Mxx += fact1 + fact2 * ex*ex + fact4 * ey*ey;
    Mxy += (fact2 - fact4)* ex*ey;
    Mxz += fact2 * ex*ez;
    Myx += (fact2 - fact4)* ex*ey;
    Myy += fact1 + fact2 * ey*ey + fact4 * ex*ex;
    Myz += fact2 * ey*ez;
    Mzx += fact2 * ez*ex + fact3 * ex;
    Mzy += fact2 * ez*ey + fact3 * ey;
    Mzz += fact1 + fact2 * ez*ez + fact3 * ez;         
  }
}

/*
 rotation_from_torque computes the product
 W = M_rt*T
*/
__global__ void rotation_from_torque(const double *x,
                                     const double *t,					
                                     double *u,
				     int number_of_blobs,
                                     double eta,
                                     double a){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;   

  double invaGPU = 1.0 / a;
  double a3 = a*a*a;

  double Ux=0;
  double Uy=0;
  double Uz=0;

  double rx, ry, rz;

  double Mxx, Mxy, Mxz;
  double Myx, Myy, Myz;
  double Mzx, Mzy, Mzz;

  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  for(int j=0; j<number_of_blobs; j++){
    joffset = j * NDIM;

    // Compute vector between particles i and j
    rx = x[ioffset    ] - x[joffset    ];
    ry = x[ioffset + 1] - x[joffset + 1];
    rz = x[ioffset + 2] - x[joffset + 2];

    // 1. Compute mobility for pair i-j
    mobilityWTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, invaGPU);
    Myx = Mxy;
    Mzx = Mxz;
    Mzy = Myz;
    mobilityWTSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy,Mzz, i,j, invaGPU, x[joffset+2]/a);

    //2. Compute product M_ij * T_j
    Ux = Ux + (Mxx * t[joffset] + Mxy * t[joffset + 1] + Mxz * t[joffset + 2]);
    Uy = Uy + (Myx * t[joffset] + Myy * t[joffset + 1] + Myz * t[joffset + 2]);
    Uz = Uz + (Mzx * t[joffset] + Mzy * t[joffset + 1] + Mzz * t[joffset + 2]);
  }
  //LOOP END

  //3. Save velocity U_i
  double pi = 4.0 * atan(1.0);
  double norm_fact_t = 8 * pi * eta * a3;
  u[ioffset    ] = Ux / norm_fact_t;
  u[ioffset + 1] = Uy / norm_fact_t;
  u[ioffset + 2] = Uz / norm_fact_t;

  return;
}

/*
 rotation_from_torque_no_wall computes the product
 W = M_rt*T
*/
__global__ void rotation_from_torque_no_wall(const double *x,
					     const double *t,					
					     double *u,
					     int number_of_blobs,
					     double eta,
					     double a){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;   

  double invaGPU = 1.0 / a;
  
  double a3 = a*a*a;

  double Ux=0;
  double Uy=0;
  double Uz=0;

  double rx, ry, rz;

  double Mxx, Mxy, Mxz;
  double Myx, Myy, Myz;
  double Mzx, Mzy, Mzz;

  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  for(int j=0; j<number_of_blobs; j++){
    joffset = j * NDIM;

    // Compute vector between particles i and j
    rx = x[ioffset    ] - x[joffset    ];
    ry = x[ioffset + 1] - x[joffset + 1];
    rz = x[ioffset + 2] - x[joffset + 2];

    // 1. Compute mobility for pair i-j
    mobilityWTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, invaGPU);
    Myx = Mxy;
    Mzx = Mxz;
    Mzy = Myz;

    //2. Compute product M_ij * T_j
    Ux = Ux + (Mxx * t[joffset] + Mxy * t[joffset + 1] + Mxz * t[joffset + 2]);
    Uy = Uy + (Myx * t[joffset] + Myy * t[joffset + 1] + Myz * t[joffset + 2]);
    Uz = Uz + (Mzx * t[joffset] + Mzy * t[joffset + 1] + Mzz * t[joffset + 2]);
  }
  //LOOP END

  //3. Save velocity U_i
  double pi = 4.0 * atan(1.0);
  double norm_fact_t = 8 * pi * eta * a3;
  u[ioffset    ] = Ux / norm_fact_t;
  u[ioffset + 1] = Uy / norm_fact_t;
  u[ioffset + 2] = Uz / norm_fact_t;

  return;
}

////////// WF //////////////////////////////////////////////////

/*
 mobilityWFRPY computes the 3x3 RPY mobility
 between blobs i and j that maps forces to
 angular velocities.
 IMPORTANT, we use the right-hand side convection,
 in the paper of Wajnryb et al. 2013 they use
 the left hand side convection!

 The mobility is normalized with 8 pi eta a**2.
*/
__device__ void mobilityWFRPY(double rx,
 			      double ry,
			      double rz,
			      double &Mxx,
			      double &Mxy,
			      double &Mxz,
			      double &Myy,
			      double &Myz,
			      double &Mzz,
			      int i,
			      int j,
                              double invaGPU){
  
  if(i==j){
    Mxx = 0;
    Mxy = 0;
    Mxz = 0;
    Myy = Mxx;
    Myz = 0;
    Mzz = Mxx;
  }
  else{
    rx = rx * invaGPU; //Normalize distance with hydrodynamic radius
    ry = ry * invaGPU;
    rz = rz * invaGPU;
    double r2 = rx*rx + ry*ry + rz*rz;
    double r = sqrt(r2);
    double r3 = r2*r;
    //We should not divide by zero but std::numeric_limits<double>::min() does not work in the GPU
    //double invr = (r > std::numeric_limits<double>::min()) ? (1.0 / r) : (1.0 / std::numeric_limits<double>::min())
    double invr3 = 1 / r3;
    double c1;
    if(r>=2){
      Mxx = 0;
      Mxy = -rz * invr3;
      Mxz =  ry * invr3;
      Myy = 0;
      Myz = -rx * invr3;
      Mzz = 0;
    }
    else{
      c1 =  0.5*( 1 - 0.375 * r); // 3/8 = 0.375
      Mxx = 0;
      Mxy = -c1 * rz;
      Mxz = c1 * ry ;
      Myy = 0;
      Myz = -c1 * rx;
      Mzz = 0;
    }
  } 
  return;
}

/*
 mobilityWFSingleWallCorrection computes the 3x3 mobility correction due to a wall
 between blobs i and j that maps forces to angular velocities.
 This uses the expression from the Swan and Brady paper for a finite size particle
 but IMPORTANT, it used the right-hand side convection, Swan's paper uses the
 left-hand side convection.

 Mobility is normalize by 8*pi*eta*a.
*/
__device__ void mobilityWFSingleWallCorrection(double rx,
			                       double ry,
			                       double rz,
			                       double &Mxx,
                  			       double &Mxy,
			                       double &Mxz,
                                               double &Myx,
			                       double &Myy,
			                       double &Myz,
                                               double &Mzx,
                                               double &Mzy,
			                       int i,
			                       int j,
                                               double invaGPU,
                                               double hj){
  if(i == j){
    double invZi = 1.0 / hj;
    double invZi4 = pow(invZi,4);
    Mxy += -invZi4 * 0.125; // 3/24 = 0.125
    Myx +=  invZi4 * 0.125; // 3/24 = 0.125
  }
  else{
    double h_hat = hj / rz;
    double invR = rsqrt(rx*rx + ry*ry + rz*rz); // = 1 / r;
    double invR2 = invR * invR;
    double invR4 = invR2 * invR2;
    double ex = rx * invR;
    double ey = ry * invR;
    double ez = rz * invR;
    
    double fact1 =  invR2;
    double fact2 = (6*h_hat*ez*ez*invR2 + (1-10*ez*ez)*invR4) * 2;
    double fact3 = -ez*(3*h_hat*invR2 - 5*invR4) * 2;
    double fact4 = -ez*(h_hat*invR2 - invR4) * 2;
    
    Mxx -=                       - fact3*ex*ey;
    Mxy -=   fact1*ez            - fact3*ey*ey + fact4;
    Mxz -= - fact1*ey - fact2*ey - fact3*ey*ez;
    Myx -= - fact1*ez            + fact3*ex*ex - fact4;
    Myy -=                         fact3*ex*ey;
    Myz -=   fact1*ex + fact2*ex + fact3*ex*ez;
    Mzx -=   fact1*ey;
    Mzy -= - fact1*ex;
  }
}


__global__ void rotation_from_force(const double *x,
                                    const double *f,					
                                    double *u,
				    int number_of_blobs,
                                    double eta,
                                    double a){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;   

  double invaGPU = 1.0 / a;
  
  double a2 = a*a;

  double Ux=0;
  double Uy=0;
  double Uz=0;

  double rx, ry, rz;

  double Mxx, Mxy, Mxz;
  double Myx, Myy, Myz;
  double Mzx, Mzy, Mzz;

  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  for(int j=0; j<number_of_blobs; j++){
    joffset = j * NDIM;

    // Compute vector between particles i and j
    rx = x[ioffset    ] - x[joffset    ];
    ry = x[ioffset + 1] - x[joffset + 1];
    rz = x[ioffset + 2] - x[joffset + 2];

    // 1. Compute mobility for pair i-j
    mobilityWFRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, invaGPU);
    Myx = -Mxy;
    Mzx = -Mxz;
    Mzy = -Myz;
    mobilityWFSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, i,j, invaGPU, x[joffset+2]/a);

    //2. Compute product M_ij * F_j
    Ux = Ux + (Mxx * f[joffset] + Mxy * f[joffset + 1] + Mxz * f[joffset + 2]);
    Uy = Uy + (Myx * f[joffset] + Myy * f[joffset + 1] + Myz * f[joffset + 2]);
    Uz = Uz + (Mzx * f[joffset] + Mzy * f[joffset + 1] + Mzz * f[joffset + 2]);
  }
  //LOOP END

  //3. Save velocity U_i
  double pi = 4.0 * atan(1.0);
  double norm_fact_t = 8 * pi * eta * a2;
  u[ioffset    ] = Ux / norm_fact_t;
  u[ioffset + 1] = Uy / norm_fact_t;
  u[ioffset + 2] = Uz / norm_fact_t;

  return;
}


__global__ void rotation_from_force_no_wall(const double *x,
					   const double *f,					
					   double *u,
					   int number_of_blobs,
					   double eta,
					   double a){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;   

  double invaGPU = 1.0 / a;
  
  double a2 = a*a;

  double Ux=0;
  double Uy=0;
  double Uz=0;

  double rx, ry, rz;

  double Mxx, Mxy, Mxz;
  double Myx, Myy, Myz;
  double Mzx, Mzy, Mzz;

  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  for(int j=0; j<number_of_blobs; j++){
    joffset = j * NDIM;

    // Compute vector between particles i and j
    rx = x[ioffset    ] - x[joffset    ];
    ry = x[ioffset + 1] - x[joffset + 1];
    rz = x[ioffset + 2] - x[joffset + 2];

    // 1. Compute mobility for pair i-j
    mobilityWFRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, invaGPU);
    Myx = -Mxy;
    Mzx = -Mxz;
    Mzy = -Myz;

    //2. Compute product M_ij * F_j
    Ux = Ux + (Mxx * f[joffset] + Mxy * f[joffset + 1] + Mxz * f[joffset + 2]);
    Uy = Uy + (Myx * f[joffset] + Myy * f[joffset + 1] + Myz * f[joffset + 2]);
    Uz = Uz + (Mzx * f[joffset] + Mzy * f[joffset + 1] + Mzz * f[joffset + 2]);
  }
  //LOOP END

  //3. Save velocity U_i
  double pi = 4.0 * atan(1.0);
  double norm_fact_t = 8 * pi * eta * a2;
  u[ioffset    ] = Ux / norm_fact_t;
  u[ioffset + 1] = Uy / norm_fact_t;
  u[ioffset + 2] = Uz / norm_fact_t;

  return;
}


////////// UT //////////////////////////////////////////////////

/*
 mobilityUTRPY computes the 3x3 RPY mobility
 between blobs i and j that maps torques to
 linear velocities.
 IMPORTANT, we use the right-hand side convection,
 in the paper of Wajnryb et al. 2013 they use
 the left hand side convection!

 The mobility is normalized with 8 pi eta a**2.
*/
__device__ void mobilityUTRPY(double rx,
			      double ry,
			      double rz,
			      double &Mxx,
			      double &Mxy,
			      double &Mxz,
			      double &Myy,
			      double &Myz,
			      double &Mzz,
			      int i,
			      int j,
                              double invaGPU){

  if(i==j){
    Mxx = 0;
    Mxy = 0;
    Mxz = 0;
    Myy = Mxx;
    Myz = 0;
    Mzz = Mxx;
  }
  else{
    rx = rx * invaGPU; //Normalize distance with hydrodynamic radius
    ry = ry * invaGPU;
    rz = rz * invaGPU;
    double r2 = rx*rx + ry*ry + rz*rz;
    double r = sqrt(r2);
    double r3 = r2*r;
    // We should not divide by zero but std::numeric_limits<double>::min() does not work in the GPU
    // double invr = (r > std::numeric_limits<double>::min()) ? (1.0 / r) : (1.0 / std::numeric_limits<double>::min())
    double invr3 = 1 / r3;
    double c1;
    if(r>=2){
      Mxx = 0;
      Mxy = rz * invr3;
      Mxz = -ry * invr3;
      Myy = 0;
      Myz = rx * invr3;
      Mzz = 0;
   
    }
    else{
      c1 = 0.5 * (1 - 0.375 * r); // 3/8 = 0.375
      Mxx = 0;
      Mxy = c1 * rz;
      Mxz = -c1 * ry ;
      Myy = 0;
      Myz = c1 * rx;
      Mzz = 0;
    }
  } 
  
  return;
}

/*
 mobilityUTSingleWallCorrection computes the 3x3 mobility correction due to a wall
 between blobs i and j that maps torques to linear velocities.
 This uses the expression from the Swan and Brady paper for a finite size particle
 but IMPORTANT, it used the right-hand side convection, Swan's paper uses the
 left-hand side convection.

 Mobility is normalize by 8*pi*eta*a.
*/

__device__ void mobilityUTSingleWallCorrection(double rx,
			                       double ry,
			                       double rz,
			                       double &Mxx,
                  			       double &Mxy,
			                       double &Mxz,
                                               double &Myx,
			                       double &Myy,
			                       double &Myz,
                                               double &Mzx,
                                               double &Mzy,
			                       int i,
			                       int j,
                                               double invaGPU,
                                               double hj){
  if(i == j){
    double invZi = 1.0 / hj;
    double invZi4 = pow(invZi,4);
    Mxy -= - invZi4 * 0.125; // 3/24 = 0.125
    Myx -=   invZi4 * 0.125; // 3/24 = 0.125
  }
  else{
    double h_hat = hj / rz;
    double invR = rsqrt(rx*rx + ry*ry + rz*rz); // = 1 / r;
    double invR2 = invR * invR;
    double invR4 = invR2 * invR2;
    double ex = rx * invR;
    double ey = ry * invR;
    double ez = rz * invR;
    
    double fact1 =  invR2;
    double fact2 = (6*h_hat*ez*ez*invR2 + (1-10*ez*ez)*invR4) * 2;
    double fact3 = -ez*(3*h_hat*invR2 - 5*invR4) * 2;
    double fact4 = -ez*(h_hat*invR2 - invR4) * 2;
    
    Mxx -=                       - fact3*ex*ey        ;
    Mxy -= - fact1*ez            + fact3*ex*ex - fact4;
    Mxz -=   fact1*ey                                 ;
    Myx -=   fact1*ez            - fact3*ey*ey + fact4;
    Myy -=                         fact3*ex*ey        ;
    Myz -= - fact1*ex                                 ;
    Mzx -= - fact1*ey - fact2*ey - fact3*ey*ez        ;
    Mzy -=   fact1*ex + fact2*ex + fact3*ex*ez        ;
  }
}


__global__ void velocity_from_force_and_torque(const double *x,
					       const double *f,
					       const double *t,
					       double *u,
					       int number_of_blobs,
					       double eta,
					       double a){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;   

  double invaGPU = 1.0 / a;
  
  double a2 = a*a;

  double Ufx=0;
  double Ufy=0;
  double Ufz=0;
  
  double Utx=0;
  double Uty=0;
  double Utz=0;

  double rx, ry, rz;

  double Mxx, Mxy, Mxz;
  double Myx, Myy, Myz;
  double Mzx, Mzy, Mzz;

  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  for(int j=0; j<number_of_blobs; j++){
    joffset = j * NDIM;

    // Compute vector between particles i and j
    rx = x[ioffset    ] - x[joffset    ];
    ry = x[ioffset + 1] - x[joffset + 1];
    rz = x[ioffset + 2] - x[joffset + 2];

    // 1. Compute UT mobility for pair i-j
    mobilityUTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, invaGPU);
    Myx = -Mxy;
    Mzx = -Mxz;
    Mzy = -Myz;
    // Mind the correct symmety! M_UT,ij^{alpha,beta} = M_WF,ji^{beta,alpha}
    // mobilityUTSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, i,j, invaGPU, x[joffset+2]/a);
    mobilityUTSingleWallCorrection(-rx/a, -ry/a, (-rz+2*x[ioffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, j,i, invaGPU, x[ioffset+2]/a);

    // 2. Compute product M_ij * T_j
    Utx = Utx + (Mxx * t[joffset] + Mxy * t[joffset + 1] + Mxz * t[joffset + 2]);
    Uty = Uty + (Myx * t[joffset] + Myy * t[joffset + 1] + Myz * t[joffset + 2]);
    Utz = Utz + (Mzx * t[joffset] + Mzy * t[joffset + 1] + Mzz * t[joffset + 2]);
    
    
    // 3. Compute UF mobility for pair i-j
    mobilityUFRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, invaGPU);
    Myx = Mxy;
    Mzx = Mxz;
    Mzy = Myz;
    mobilityUFSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy,Mzz, i,j, invaGPU, x[joffset+2]/a);
    
    // 4. Compute product M_ij * F_j
    Ufx = Ufx + (Mxx * f[joffset] + Mxy * f[joffset + 1] + Mxz * f[joffset + 2]);
    Ufy = Ufy + (Myx * f[joffset] + Myy * f[joffset + 1] + Myz * f[joffset + 2]);
    Ufz = Ufz + (Mzx * f[joffset] + Mzy * f[joffset + 1] + Mzz * f[joffset + 2]);
  }
  //LOOP END

  //3. Save velocity U_i
  double pi = 4.0 * atan(1.0);
  double norm_fact_t = 8 * pi * eta * a2;
  double norm_fact_f = 8 * pi * eta * a;
  u[ioffset    ] = Utx / norm_fact_t + Ufx / norm_fact_f;
  u[ioffset + 1] = Uty / norm_fact_t + Ufy / norm_fact_f;
  u[ioffset + 2] = Utz / norm_fact_t + Ufz / norm_fact_f;

  return;
}


__global__ void velocity_from_force_and_torque_no_wall(const double *x,
						       const double *f,
						       const double *t,
						       double *u,
						       int number_of_blobs,
						       double eta,
						       double a){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;   

  double invaGPU = 1.0 / a;
  
  double a2 = a*a;

  double Ufx=0;
  double Ufy=0;
  double Ufz=0;
  
  double Utx=0;
  double Uty=0;
  double Utz=0;

  double rx, ry, rz;

  double Mxx, Mxy, Mxz;
  double Myx, Myy, Myz;
  double Mzx, Mzy, Mzz;

  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  for(int j=0; j<number_of_blobs; j++){
    joffset = j * NDIM;

    // Compute vector between particles i and j
    rx = x[ioffset    ] - x[joffset    ];
    ry = x[ioffset + 1] - x[joffset + 1];
    rz = x[ioffset + 2] - x[joffset + 2];

    // 1. Compute UT mobility for pair i-j
    mobilityUTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, invaGPU);
    Myx = -Mxy;
    Mzx = -Mxz;
    Mzy = -Myz;

    // 2. Compute product M_ij * T_j
    Utx = Utx + (Mxx * t[joffset] + Mxy * t[joffset + 1] + Mxz * t[joffset + 2]);
    Uty = Uty + (Myx * t[joffset] + Myy * t[joffset + 1] + Myz * t[joffset + 2]);
    Utz = Utz + (Mzx * t[joffset] + Mzy * t[joffset + 1] + Mzz * t[joffset + 2]);
    
    
    // 3. Compute UF mobility for pair i-j
    mobilityUFRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, invaGPU);
    Myx = Mxy;
    Mzx = Mxz;
    Mzy = Myz;
    
    // 4. Compute product M_ij * F_j
    Ufx = Ufx + (Mxx * f[joffset] + Mxy * f[joffset + 1] + Mxz * f[joffset + 2]);
    Ufy = Ufy + (Myx * f[joffset] + Myy * f[joffset + 1] + Myz * f[joffset + 2]);
    Ufz = Ufz + (Mzx * f[joffset] + Mzy * f[joffset + 1] + Mzz * f[joffset + 2]);
  }
  //LOOP END

  //3. Save velocity U_i
  double pi = 4.0 * atan(1.0);
  double norm_fact_t = 8 * pi * eta * a2;
  double norm_fact_f = 8 * pi * eta * a;
  u[ioffset    ] = Utx / norm_fact_t + Ufx / norm_fact_f;
  u[ioffset + 1] = Uty / norm_fact_t + Ufy / norm_fact_f;
  u[ioffset + 2] = Utz / norm_fact_t + Ufz / norm_fact_f;

  return;
}


__global__ void velocity_from_torque(const double *x,
 			             const double *t,
				     double *u,
				     int number_of_blobs,
				     double eta,
				     double a){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;   

  double invaGPU = 1.0 / a;
  
  double a2 = a*a;

  double Utx=0;
  double Uty=0;
  double Utz=0;

  double rx, ry, rz;

  double Mxx, Mxy, Mxz;
  double Myx, Myy, Myz;
  double Mzx, Mzy, Mzz;

  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  for(int j=0; j<number_of_blobs; j++){
    joffset = j * NDIM;

    // Compute vector between particles i and j
    rx = x[ioffset    ] - x[joffset    ];
    ry = x[ioffset + 1] - x[joffset + 1];
    rz = x[ioffset + 2] - x[joffset + 2];

    // 1. Compute UT mobility for pair i-j
    mobilityUTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, invaGPU);
    Myx = -Mxy;
    Mzx = -Mxz;
    Mzy = -Myz;
    // Mind the correct symmety! M_UT,ij^{alpha,beta} = M_WF,ji^{beta,alpha}
    // mobilityUTSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, i,j, invaGPU, x[joffset+2]/a);
    mobilityUTSingleWallCorrection(-rx/a, -ry/a, (-rz+2*x[ioffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, j,i, invaGPU, x[ioffset+2]/a);

    // 2. Compute product M_ij * T_j
    Utx = Utx + (Mxx * t[joffset] + Mxy * t[joffset + 1] + Mxz * t[joffset + 2]);
    Uty = Uty + (Myx * t[joffset] + Myy * t[joffset + 1] + Myz * t[joffset + 2]);
    Utz = Utz + (Mzx * t[joffset] + Mzy * t[joffset + 1] + Mzz * t[joffset + 2]);
  }
  //LOOP END

  //3. Save velocity U_i
  double pi = 4.0 * atan(1.0);
  double norm_fact_t = 1.0 / (8 * pi * eta * a2);
  u[ioffset    ] = Utx * norm_fact_t ;
  u[ioffset + 1] = Uty * norm_fact_t ;
  u[ioffset + 2] = Utz * norm_fact_t ;

  return;
}


////////// UF - single precision //////////////////////////////////////////////////

__device__ void mobilityUFRPY_single(float rx,
			             float ry,
			             float rz,
			             float &Mxx,
			             float &Mxy,
			             float &Mxz,
			             float &Myy,
			             float &Myz,
			             float &Mzz,
			             int i,
			             int j,
                                     float invaGPU){
  
  float fourOverThree = 4.0 / 3.0;

  if(i==j){
    Mxx = fourOverThree;
    Mxy = 0;
    Mxz = 0;
    Myy = Mxx;
    Myz = 0;
    Mzz = Mxx;
  }
  else{
    rx = rx * invaGPU; //Normalize distance with hydrodynamic radius
    ry = ry * invaGPU;
    rz = rz * invaGPU;
    float r2 = rx*rx + ry*ry + rz*rz;
    float r = sqrt(r2);
    //We should not divide by zero but std::numeric_limits<float>::min() does not work in the GPU
    //float invr = (r > std::numeric_limits<float>::min()) ? (1.0 / r) : (1.0 / std::numeric_limits<float>::min())
    float invr = 1.0 / r;
    float invr2 = invr * invr;
    float c1, c2;
    if(r>=2){
      c1 = 1 + 2 / (3 * r2);
      c2 = (1 - 2 * invr2) * invr2;
      Mxx = (c1 + c2*rx*rx) * invr;
      Mxy = (     c2*rx*ry) * invr;
      Mxz = (     c2*rx*rz) * invr;
      Myy = (c1 + c2*ry*ry) * invr;
      Myz = (     c2*ry*rz) * invr;
      Mzz = (c1 + c2*rz*rz) * invr;
    }
    else{
      c1 = fourOverThree * (1 - 0.28125 * r); // 9/32 = 0.28125
      c2 = fourOverThree * 0.09375 * invr;    // 3/32 = 0.09375
      Mxx = c1 + c2 * rx*rx ;
      Mxy =      c2 * rx*ry ;
      Mxz =      c2 * rx*rz ;
      Myy = c1 + c2 * ry*ry ;
      Myz =      c2 * ry*rz ;
      Mzz = c1 + c2 * rz*rz ;
    }
  } 
  
  return;
}



__device__ void mobilityUFSingleWallCorrection_single(float rx,
			                              float ry,
			                              float rz,
			                              float &Mxx,
                  			              float &Mxy,
			                              float &Mxz,
                                                      float &Myx,
			                              float &Myy,
			                              float &Myz,
                                                      float &Mzx,
                                                      float &Mzy,
			                              float &Mzz,
			                              int i,
			                              int j,
                                                      float invaGPU,
                                                      float hj){
  if(i == j){
    float invZi = 1.0 / hj;
    Mxx += -(9*invZi - 2*pow(invZi,3) + pow(invZi,5)) / 12.0;
    Myy += -(9*invZi - 2*pow(invZi,3) + pow(invZi,5)) / 12.0;
    Mzz += -(9*invZi - 4*pow(invZi,3) + pow(invZi,5)) / 6.0;
  }
  else{
    float h_hat = hj / rz;
    float invR = rsqrt(rx*rx + ry*ry + rz*rz); // = 1 / r;
    float ex = rx * invR;
    float ey = ry * invR;
    float ez = rz * invR;
    
    float fact1 = -(3*(1+2*h_hat*(1-h_hat)*ez*ez) * invR + 2*(1-3*ez*ez) * pow(invR,3) - 2*(1-5*ez*ez) * pow(invR,5))  / 3.0;
    float fact2 = -(3*(1-6*h_hat*(1-h_hat)*ez*ez) * invR - 6*(1-5*ez*ez) * pow(invR,3) + 10*(1-7*ez*ez) * pow(invR,5)) / 3.0;
    float fact3 =  ez * (3*h_hat*(1-6*(1-h_hat)*ez*ez) * invR - 6*(1-5*ez*ez) * pow(invR,3) + 10*(2-7*ez*ez) * pow(invR,5)) * 2.0 / 3.0;
    float fact4 =  ez * (3*h_hat*invR - 10*pow(invR,5)) * 2.0 / 3.0;
    float fact5 = -(3*h_hat*h_hat*ez*ez*invR + 3*ez*ez*pow(invR, 3) + (2-15*ez*ez)*pow(invR, 5)) * 4.0 / 3.0;
    
    Mxx += fact1 + fact2 * ex*ex;
    Mxy += fact2 * ex*ey;
    Mxz += fact2 * ex*ez + fact3 * ex;
    Myx += fact2 * ey*ex;
    Myy += fact1 + fact2 * ey*ey;
    Myz += fact2 * ey*ez + fact3 * ey;
    Mzx += fact2 * ez*ex + fact4 * ex;
    Mzy += fact2 * ez*ey + fact4 * ey;
    Mzz += fact1 + fact2 * ez*ez + fact3 * ez + fact4 * ez + fact5;         
  }

}


__global__ void velocity_from_force_single(const float *x,
                                           const float *f,					
                                           float *u,
				           int number_of_blobs,
                                           float eta,
                                           float a){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;   

  float invaGPU = 1.0 / a;

  float Ux=0;
  float Uy=0;
  float Uz=0;

  float rx, ry, rz;

  float Mxx, Mxy, Mxz;
  float Myx, Myy, Myz;
  float Mzx, Mzy, Mzz;

  int NDIM = 3; // 3 is the spatial dimension
  int ioffset = i * NDIM; 
  int joffset;
  
  for(int j=0; j<number_of_blobs; j++){
    joffset = j * NDIM;

    // Compute vector between particles i and j
    rx = x[ioffset    ] - x[joffset    ];
    ry = x[ioffset + 1] - x[joffset + 1];
    rz = x[ioffset + 2] - x[joffset + 2];

    // 1. Compute mobility for pair i-j
    mobilityUFRPY_single(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, invaGPU);
    Myx = Mxy;
    Mzx = Mxz;
    Mzy = Myz;
    mobilityUFSingleWallCorrection_single(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy,Mzz, i,j, invaGPU, x[joffset+2]/a);

    //2. Compute product M_ij * F_j
    Ux = Ux + (Mxx * f[joffset] + Mxy * f[joffset + 1] + Mxz * f[joffset + 2]);
    Uy = Uy + (Myx * f[joffset] + Myy * f[joffset + 1] + Myz * f[joffset + 2]);
    Uz = Uz + (Mzx * f[joffset] + Mzy * f[joffset + 1] + Mzz * f[joffset + 2]);
  }
  //LOOP END

  //3. Save velocity U_i
  float pi = 4.0 * atan(1.0);
  u[ioffset    ] = Ux / (8 * pi * eta * a);
  u[ioffset + 1] = Uy / (8 * pi * eta * a);
  u[ioffset + 2] = Uz / (8 * pi * eta * a);

  return;
}

""")

def set_number_of_threads_and_blocks(num_elements):
  '''
  This functions uses a heuristic method to determine
  the number of blocks and threads per block to be
  used in CUDA kernels.
  '''
  threads_per_block=512
  if((num_elements/threads_per_block) < 512):
    threads_per_block = 256
  if((num_elements/threads_per_block) < 256):
    threads_per_block = 128
  if((num_elements/threads_per_block) < 128):
    threads_per_block = 64
  if((num_elements/threads_per_block) < 128):
    threads_per_block = 32
  num_blocks = (num_elements-1)/threads_per_block + 1

  return (threads_per_block, num_blocks)


def single_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):
   
  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = np.reshape(r_vectors, number_of_blobs * 3)
        
  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  f_gpu = cuda.mem_alloc(force.nbytes)
  u_gpu = cuda.mem_alloc(force.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)
    
  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(f_gpu, force)
    
  # Get mobility function
  mobility = mod.get_function("velocity_from_force")

  # Compute mobility force product
  mobility(x_gpu, f_gpu, u_gpu, number_of_blobs, np.float64(eta), np.float64(a), np.float64(L[0]), np.float64(L[1]), np.float64(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1)) 
    
  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(force)
  cuda.memcpy_dtoh(u, u_gpu)
  return u

def single_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a):
   
  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Reshape arrays
  x = np.reshape(r_vectors, number_of_blobs * 3) 
    
  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  f_gpu = cuda.mem_alloc(force.nbytes)
  u_gpu = cuda.mem_alloc(force.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)
    
  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(f_gpu, force)
    
  # Get mobility function
  mobility = mod.get_function("rotation_from_force")
  
  # Compute mobility force product
  mobility(x_gpu, f_gpu, u_gpu, number_of_blobs, np.float64(eta), np.float64(a), block=(threads_per_block, 1, 1), grid=(num_blocks, 1)) 
    
  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(force)
  cuda.memcpy_dtoh(u, u_gpu)

  return u
  
def no_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a):
  
  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)
  
  # Reshape arrays
  x = np.reshape(r_vectors, number_of_blobs * 3)    
        
  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  f_gpu = cuda.mem_alloc(force.nbytes)
  u_gpu = cuda.mem_alloc(force.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)
    
  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(f_gpu, force)
    
  # Get mobility function
  mobility = mod.get_function("rotation_from_force_no_wall")

  # Compute mobility force product
  mobility(x_gpu, f_gpu, u_gpu, number_of_blobs, np.float64(eta), np.float64(a), block=(threads_per_block, 1, 1), grid=(num_blocks, 1)) 
    
  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(force)
  cuda.memcpy_dtoh(u, u_gpu)

  return u
  
def single_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a):
   
  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Reshape arrays
  x = np.reshape(r_vectors, number_of_blobs * 3)
    
  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  t_gpu = cuda.mem_alloc(torque.nbytes)
  u_gpu = cuda.mem_alloc(torque.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)
  
  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(t_gpu, torque)
    
  # Get mobility function
  mobility = mod.get_function("rotation_from_torque")

  # Compute mobility force product
  mobility(x_gpu, t_gpu, u_gpu, number_of_blobs, np.float64(eta), np.float64(a), block=(threads_per_block, 1, 1), grid=(num_blocks, 1)) 
    
  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(torque)
  cuda.memcpy_dtoh(u, u_gpu)

  return u  

def no_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a):
   
  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Reshape arrays
  x = np.reshape(r_vectors, number_of_blobs * 3)
        
  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  t_gpu = cuda.mem_alloc(torque.nbytes)
  u_gpu = cuda.mem_alloc(torque.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)
    
  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(t_gpu, torque)
    
  # Get mobility function
  mobility = mod.get_function("rotation_from_torque_no_wall")

  # Compute mobility force product
  mobility(x_gpu, t_gpu, u_gpu, number_of_blobs, np.float64(eta), np.float64(a), block=(threads_per_block, 1, 1), grid=(num_blocks, 1)) 
    
  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(torque)
  cuda.memcpy_dtoh(u, u_gpu)

  return u
  
  
def single_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force, torque, eta, a):
   
  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Reshape arrays
  x = np.reshape(r_vectors, number_of_blobs * 3)
        
  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  t_gpu = cuda.mem_alloc(torque.nbytes)
  f_gpu = cuda.mem_alloc(force.nbytes)
  u_gpu = cuda.mem_alloc(torque.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)
    
  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(t_gpu, torque)
  cuda.memcpy_htod(f_gpu, force)
    
  # Get mobility function
  mobility = mod.get_function("velocity_from_force_and_torque")

  # Compute mobility force product
  mobility(x_gpu, f_gpu, t_gpu, u_gpu, number_of_blobs, np.float64(eta), np.float64(a), block=(threads_per_block, 1, 1), grid=(num_blocks, 1)) 
    
  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(torque)
  cuda.memcpy_dtoh(u, u_gpu)

  return u  
  
def no_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force, torque, eta, a):
  
  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)
  
  # Reshape arrays
  x = np.reshape(r_vectors, number_of_blobs * 3)
           
  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  t_gpu = cuda.mem_alloc(torque.nbytes)
  f_gpu = cuda.mem_alloc(force.nbytes)
  u_gpu = cuda.mem_alloc(torque.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)
    
  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(t_gpu, torque)
  cuda.memcpy_htod(f_gpu, force)
  
  # Get mobility function
  mobility = mod.get_function("velocity_from_force_and_torque_no_wall")

  # Compute mobility force product
  mobility(x_gpu, f_gpu, t_gpu, u_gpu, number_of_blobs, np.float64(eta), np.float64(a), block=(threads_per_block, 1, 1), grid=(num_blocks, 1)) 
    
  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(torque)
  cuda.memcpy_dtoh(u, u_gpu)

  return u   
  
def single_wall_mobility_trans_times_force_pycuda_single(r_vectors, force, eta, a):
   
  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Reshape arrays
  x = np.float32(np.reshape(r_vectors, number_of_blobs * 3))
  f = np.float32(np.reshape(force, number_of_blobs * 3))
              
  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  f_gpu = cuda.mem_alloc(f.nbytes)
  u_gpu = cuda.mem_alloc(f.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)
    
  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(f_gpu, f)
    
  # Get mobility function
  mobility = mod.get_function("velocity_from_force_single")

  # Compute mobility force product
  mobility(x_gpu, f_gpu, u_gpu, number_of_blobs, np.float32(eta), np.float32(a), block=(threads_per_block, 1, 1), grid=(num_blocks, 1)) 
    
  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(f)
  cuda.memcpy_dtoh(u, u_gpu)

  return u  


def single_wall_mobility_trans_times_torque_pycuda(r_vectors, torque, eta, a):
   
  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Reshape arrays
  x = np.reshape(r_vectors, number_of_blobs * 3)
        
  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  t_gpu = cuda.mem_alloc(torque.nbytes)
  u_gpu = cuda.mem_alloc(torque.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)
    
  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(t_gpu, torque)
    
  # Get mobility function
  mobility = mod.get_function("velocity_from_torque")

  # Compute mobility force product
  mobility(x_gpu, t_gpu, u_gpu, number_of_blobs, np.float64(eta), np.float64(a), block=(threads_per_block, 1, 1), grid=(num_blocks, 1)) 
    
  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(torque)
  cuda.memcpy_dtoh(u, u_gpu)

  return u  
