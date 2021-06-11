
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# These lines set the precision of the cuda code
# to single or double. Set the precision
# in the following lines and edit the lines
# after   'mod = SourceModule("""'    accordingly
#precision = 'single'
precision = 'double'

mod = SourceModule("""
// Set real to single or double precision.
// This value has to agree witht the value
// for precision setted above.
//typedef float real;
typedef double real;


#include <stdio.h>
/*
 mobilityUFRPY computes the 3x3 RPY mobility
 between blobs i and j normalized with 8 pi eta a
*/
__device__ void mobilityUFRPY(real rx,
			      real ry,
			      real rz,
			      real &Mxx,
			      real &Mxy,
			      real &Mxz,
			      real &Myy,
			      real &Myz,
			      real &Mzz,
			      int i,
                              int j,
                              real invaGPU){

  real fourOverThree = real(4.0) / real(3.0);

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
    real r2 = rx*rx + ry*ry + rz*rz;
    real r = sqrt(r2);
    //We should not divide by zero but std::numeric_limits<real>::min() does not work in the GPU
    //real invr = (r > std::numeric_limits<real>::min()) ? (real(1.0) / r) : (real(1.0) / std::numeric_limits<real>::min())
    real invr = real(1.0) / r;
    real invr2 = invr * invr;
    real c1, c2;
    if(r>=2){
      c1 = real(1.0) + real(2.0) / (real(3.0) * r2);
      c2 = (real(1.0) - real(2.0) * invr2) * invr2;
      Mxx = (c1 + c2*rx*rx) * invr;
      Mxy = (     c2*rx*ry) * invr;
      Mxz = (     c2*rx*rz) * invr;
      Myy = (c1 + c2*ry*ry) * invr;
      Myz = (     c2*ry*rz) * invr;
      Mzz = (c1 + c2*rz*rz) * invr;
    }
    else{
      c1 = fourOverThree * (real(1.0) - real(0.28125) * r); // 9/32 = 0.28125
      c2 = fourOverThree * real(0.09375) * invr;    // 3/32 = 0.09375
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
__device__ void mobilityUFSingleWallCorrection(real rx,
			                       real ry,
			                       real rz,
			                       real &Mxx,
                  			       real &Mxy,
			                       real &Mxz,
                                               real &Myx,
			                       real &Myy,
			                       real &Myz,
                                               real &Mzx,
                                               real &Mzy,
			                       real &Mzz,
			                       int i,
                                               int j,
                                               real invaGPU,
                                               real hj){

  if(i == j){
    real invZi = real(1.0) / hj;
    real invZi3 = invZi * invZi * invZi;
    real invZi5 = invZi3 * invZi * invZi;
    Mxx += -(9*invZi - 2*invZi3 + invZi5 ) / real(12.0);
    Myy += -(9*invZi - 2*invZi3 + invZi5 ) / real(12.0);
    Mzz += -(9*invZi - 4*invZi3 + invZi5 ) / real(6.0);
  }
  else{
    real h_hat = hj / rz;
    real invR = rsqrt(rx*rx + ry*ry + rz*rz); // = 1 / r;
    real ex = rx * invR;
    real ey = ry * invR;
    real ez = rz * invR;
    real invR3 = invR * invR * invR;
    real invR5 = invR3 * invR * invR;

    real fact1 = -(3*(1+2*h_hat*(1-h_hat)*ez*ez) * invR + 2*(1-3*ez*ez) * invR3 - 2*(1-5*ez*ez) * invR5)  / real(3.0);
    real fact2 = -(3*(1-6*h_hat*(1-h_hat)*ez*ez) * invR - 6*(1-5*ez*ez) * invR3 + 10*(1-7*ez*ez) * invR5) / real(3.0);
    real fact3 =  ez * (3*h_hat*(1-6*(1-h_hat)*ez*ez) * invR - 6*(1-5*ez*ez) * invR3 + 10*(2-7*ez*ez) * invR5) * real(2.0) / real(3.0);
    real fact4 =  ez * (3*h_hat*invR - 10*invR5) * real(2.0) / real(3.0);
    real fact5 = -(3*h_hat*h_hat*ez*ez*invR + 3*ez*ez*invR3 + (2-15*ez*ez)*invR5) * real(4.0) / real(3.0);

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
__global__ void velocity_from_force(const real *x,
                                    const real *f,
                                    real *u,
				    int number_of_blobs,
                                    real eta,
                                    real a,
                                    real Lx,
                                    real Ly,
                                    real Lz){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;

  real invaGPU = real(1.0) / a;

  real Ux=0;
  real Uy=0;
  real Uz=0;

  real rx, ry, rz;

  real Mxx, Mxy, Mxz;
  real Myx, Myy, Myz;
  real Mzx, Mzy, Mzz;

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
	    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
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
  real pi = real(4.0) * atan(real(1.0));
  real norm_fact_f = real(1.0) / (8 * pi * eta * a);
  u[ioffset    ] = Ux * norm_fact_f;
  u[ioffset + 1] = Uy * norm_fact_f;
  u[ioffset + 2] = Uz * norm_fact_f;

  return;
}

/*
 velocity_from_force_in_plane computes the product
 U = M*F. TODO:This is computing too much, change to not compute Mzz and the like
*/
__global__ void velocity_from_force_in_plane(const real *x,
                                    const real *f,
                                    real *u,
				    int number_of_blobs,
                                    real eta,
                                    real a,
                                    real Lx,
                                    real Ly,
                                    real Lz){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;

  real invaGPU = real(1.0) / a;

  real Ux=0;
  real Uy=0;

  real rx, ry, rz;

  real Mxx, Mxy, Mxz;
  real Myx, Myy, Myz;
  real Mzx, Mzy, Mzz;

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
	    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
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
	  Ux = Ux + (Mxx * f[joffset] + Mxy * f[joffset + 1]); //  + Mxz * f[joffset + 2]
	  Uy = Uy + (Myx * f[joffset] + Myy * f[joffset + 1]); //  + Myz * f[joffset + 2]
	}
      }
    }
  }
  //LOOP END

  //3. Save velocity U_i
  real pi = real(4.0) * atan(real(1.0));
  real norm_fact_f = real(1.0) / (8 * pi * eta * a);
  u[ioffset    ] = Ux * norm_fact_f;
  u[ioffset + 1] = Uy * norm_fact_f;
  u[ioffset + 2] = 0;

  return;
}




/*
 velocity_from_force computes the product
 U = M*F
*/
__global__ void velocity_from_force_no_wall(const real *x,
                                            const real *f,
                                            real *u,
				            int number_of_blobs,
                                            real eta,
                                            real a,
                                            real Lx,
                                            real Ly,
                                            real Lz){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;

  real invaGPU = real(1.0) / a;

  real Ux=0;
  real Uy=0;
  real Uz=0;

  real rx, ry, rz;

  real Mxx, Mxy, Mxz;
  real Myy, Myz;
  real Mzz;

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
	  rx = x[ioffset    ] - x[joffset    ];
	  ry = x[ioffset + 1] - x[joffset + 1];
	  rz = x[ioffset + 2] - x[joffset + 2];

	  // Project a vector r to the extended unit cell
	  // centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If
	  // any dimension of L is equal or smaller than zero the
	  // box is assumed to be infinite in that direction.
	  if(Lx > 0){
	    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
            rz = rz + boxZ * Lz;
	  }

	  // 1. Compute mobility for pair i-j, if i==j use self-interation
          int j_image = j;
          if(boxX!=0 or boxY!=0 or boxZ!=0){
            j_image = -1;
          }
	  mobilityUFRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);

	  //2. Compute product M_ij * F_j
	  Ux = Ux + (Mxx * f[joffset] + Mxy * f[joffset + 1] + Mxz * f[joffset + 2]);
	  Uy = Uy + (Mxy * f[joffset] + Myy * f[joffset + 1] + Myz * f[joffset + 2]);
	  Uz = Uz + (Mxz * f[joffset] + Myz * f[joffset + 1] + Mzz * f[joffset + 2]);
	}
      }
    }
  }
  //LOOP END

  //3. Save velocity U_i
  real pi = real(4.0) * atan(real(1.0));
  real norm_fact_f = real(1.0) / (8 * pi * eta * a);
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
__device__ void mobilityWTRPY(real rx,
			      real ry,
			      real rz,
			      real &Mxx,
			      real &Mxy,
			      real &Mxz,
			      real &Myy,
			      real &Myz,
			      real &Mzz,
			      int i,
			      int j,
                              real invaGPU){

  if(i==j){
    Mxx = real(1.0);
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
    real r2 = rx*rx + ry*ry + rz*rz;
    real r = sqrt(r2);
    real r3 = r2*r;
    //We should not divide by zero but std::numeric_limits<real>::min() does not work in the GPU
    //real invr = (r > std::numeric_limits<real>::min()) ? (1.0 / r) : (1.0 / std::numeric_limits<real>::min())
    real invr = real(1.0) / r;
    real invr2 = real(1.0) / r2;
    real invr3 = real(1.0) / r3;
    real c1, c2;
    if(r>=2){
      c1 = -real(0.5);
      c2 = real(1.5) * invr2 ;
      Mxx = (c1 + c2*rx*rx) * invr3;
      Mxy = (     c2*rx*ry) * invr3;
      Mxz = (     c2*rx*rz) * invr3;
      Myy = (c1 + c2*ry*ry) * invr3;
      Myz = (     c2*ry*rz) * invr3;
      Mzz = (c1 + c2*rz*rz) * invr3;
    }
    else{
      c1 =  (real(1.0) - real(0.84375) * r + real(0.078125) * r3); // 27/32 = 0.84375, 5/64 = 0.078125
      c2 =  real(0.28125) * invr - real(0.046875) * r;    // 9/32 = 0.28125, 3/64 = 0.046875
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
__device__ void mobilityWTSingleWallCorrection(real rx,
			                       real ry,
			                       real rz,
			                       real &Mxx,
                  			       real &Mxy,
			                       real &Mxz,
                                               real &Myx,
			                       real &Myy,
			                       real &Myz,
                                               real &Mzx,
                                               real &Mzy,
			                       real &Mzz,
			                       int i,
			                       int j,
                                               real invaGPU,
                                               real hj){
  if(i == j){
    real invZi = real(1.0) / hj;
    real invZi3 = pow(invZi,3);
    Mxx += - invZi3 * real(0.3125); // 15/48 = 0.3125
    Myy += - invZi3 * real(0.3125); // 15/48 = 0.3125
    Mzz += - invZi3 * real(0.125); // 3/24 = 0.125
  }
  else{
    real invR = rsqrt(rx*rx + ry*ry + rz*rz); // = 1 / r;
    real invR3 = invR * invR * invR;
    real ex = rx * invR;
    real ey = ry * invR;
    real ez = rz * invR;

    real fact1 =  ((1-6*ez*ez) * invR3 ) / real(2.0);
    real fact2 = -(9 * invR3) / real(6.0);
    real fact3 =  (3 * invR3 * ez);
    real fact4 =  (3 * invR3);

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
__global__ void rotation_from_torque(const real *x,
                                     const real *t,
                                     real *u,
				     int number_of_blobs,
                                     real eta,
                                     real a,
                                     real Lx,
                                     real Ly,
                                     real Lz){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;

  real invaGPU = real(1.0) / a;
  real a3 = a*a*a;

  real Ux=0;
  real Uy=0;
  real Uz=0;

  real rx, ry, rz;

  real Mxx, Mxy, Mxz;
  real Myx, Myy, Myz;
  real Mzx, Mzy, Mzz;

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
          rx = x[ioffset    ] - x[joffset    ];
          ry = x[ioffset + 1] - x[joffset + 1];
          rz = x[ioffset + 2] - x[joffset + 2];

	  // Project a vector r to the extended unit cell
	  // centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If
	  // any dimension of L is equal or smaller than zero the
	  // box is assumed to be infinite in that direction.
	  if(Lx > 0){
	    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
            rz = rz + boxZ * Lz;
	  }

	  // 1. Compute mobility for pair i-j, if i==j use self-interation
          int j_image = j;
          if(boxX!=0 or boxY!=0 or boxZ!=0){
            j_image = -1;
          }

          // 1. Compute mobility for pair i-j
          mobilityWTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
          Myx = Mxy;
          Mzx = Mxz;
          Mzy = Myz;
          mobilityWTSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy,Mzz, i,j_image, invaGPU, x[joffset+2]/a);

          //2. Compute product M_ij * T_j
          Ux = Ux + (Mxx * t[joffset] + Mxy * t[joffset + 1] + Mxz * t[joffset + 2]);
          Uy = Uy + (Myx * t[joffset] + Myy * t[joffset + 1] + Myz * t[joffset + 2]);
          Uz = Uz + (Mzx * t[joffset] + Mzy * t[joffset + 1] + Mzz * t[joffset + 2]);
        }
      }
    }
  }
  //LOOP END

  //3. Save velocity U_i
  real pi = real(4.0) * atan(real(1.0));
  real norm_fact_t = 8 * pi * eta * a3;
  u[ioffset    ] = Ux / norm_fact_t;
  u[ioffset + 1] = Uy / norm_fact_t;
  u[ioffset + 2] = Uz / norm_fact_t;

  return;
}

/*
 rotation_from_torque_no_wall computes the product
 W = M_rt*T
*/
__global__ void rotation_from_torque_no_wall(const real *x,
					     const real *t,
					     real *u,
					     int number_of_blobs,
					     real eta,
					     real a,
                                             real Lx,
                                             real Ly,
                                             real Lz){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;

  real invaGPU = real(1.0) / a;

  real a3 = a*a*a;

  real Ux=0;
  real Uy=0;
  real Uz=0;

  real rx, ry, rz;

  real Mxx, Mxy, Mxz;
  real Myx, Myy, Myz;
  real Mzx, Mzy, Mzz;

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
          rx = x[ioffset    ] - x[joffset    ];
          ry = x[ioffset + 1] - x[joffset + 1];
          rz = x[ioffset + 2] - x[joffset + 2];

	  // Project a vector r to the extended unit cell
	  // centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If
	  // any dimension of L is equal or smaller than zero the
	  // box is assumed to be infinite in that direction.
	  if(Lx > 0){
	    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
            rz = rz + boxZ * Lz;
	  }

	  // 1. Compute mobility for pair i-j, if i==j use self-interation
          int j_image = j;
          if(boxX!=0 or boxY!=0 or boxZ!=0){
            j_image = -1;
          }

          // 1. Compute mobility for pair i-j
          mobilityWTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
          Myx = Mxy;
          Mzx = Mxz;
          Mzy = Myz;

          //2. Compute product M_ij * T_j
          Ux = Ux + (Mxx * t[joffset] + Mxy * t[joffset + 1] + Mxz * t[joffset + 2]);
          Uy = Uy + (Myx * t[joffset] + Myy * t[joffset + 1] + Myz * t[joffset + 2]);
          Uz = Uz + (Mzx * t[joffset] + Mzy * t[joffset + 1] + Mzz * t[joffset + 2]);
        }
      }
    }
  }
  //LOOP END

  //3. Save velocity U_i
  real pi = real(4.0) * atan(real(1.0));
  real norm_fact_t = 8 * pi * eta * a3;
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

 The mobility is normalized with 8 pi eta a**2.
*/
__device__ void mobilityWFRPY(real rx,
 			      real ry,
			      real rz,
			      real &Mxx,
			      real &Mxy,
			      real &Mxz,
			      real &Myy,
			      real &Myz,
			      real &Mzz,
			      int i,
			      int j,
                              real invaGPU){

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
    real r2 = rx*rx + ry*ry + rz*rz;
    real r = sqrt(r2);
    real r3 = r2*r;
    //We should not divide by zero but std::numeric_limits<real>::min() does not work in the GPU
    //real invr = (r > std::numeric_limits<real>::min()) ? (1.0 / r) : (1.0 / std::numeric_limits<real>::min())
    real invr3 = real(1.0) / r3;
    real c1;
    if(r>=2){
      Mxx =  0;
      Mxy =  rz * invr3;
      Mxz = -ry * invr3;
      Myy =  0;
      Myz =  rx * invr3;
      Mzz =  0;
    }
    else{
      c1 =  real(0.5)*( real(1.0) - real(0.375) * r); // 3/8 = 0.375
      Mxx =  0;
      Mxy =  c1 * rz;
      Mxz = -c1 * ry ;
      Myy =  0;
      Myz =  c1 * rx;
      Mzz =  0;
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
__device__ void mobilityWFSingleWallCorrection(real rx,
			                       real ry,
			                       real rz,
			                       real &Mxx,
                  			       real &Mxy,
			                       real &Mxz,
                                               real &Myx,
			                       real &Myy,
			                       real &Myz,
                                               real &Mzx,
                                               real &Mzy,
			                       int i,
			                       int j,
                                               real invaGPU,
                                               real hj){
  if(i == j){
    real invZi = real(1.0) / hj;
    real invZi4 = pow(invZi,4);
    Mxy += -invZi4 * real(0.125); // 3/24 = 0.125
    Myx +=  invZi4 * real(0.125); // 3/24 = 0.125
  }
  else{
    real h_hat = hj / rz;
    real invR = rsqrt(rx*rx + ry*ry + rz*rz); // = 1 / r;
    real invR2 = invR * invR;
    real invR4 = invR2 * invR2;
    real ex = rx * invR;
    real ey = ry * invR;
    real ez = rz * invR;

    real fact1 =  invR2;
    real fact2 = (6*h_hat*ez*ez*invR2 + (1-10*ez*ez)*invR4) * real(2.0);
    real fact3 = -ez*(3*h_hat*invR2 - 5*invR4) * real(2.0);
    real fact4 = -ez*(h_hat*invR2 - invR4) * real(2.0);

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


__global__ void rotation_from_force(const real *x,
                                    const real *f,
                                    real *u,
				    int number_of_blobs,
                                    real eta,
                                    real a,
                                    real Lx,
                                    real Ly,
                                    real Lz){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;

  real invaGPU = real(1.0) / a;

  real a2 = a*a;

  real Ux=0;
  real Uy=0;
  real Uz=0;

  real rx, ry, rz;

  real Mxx, Mxy, Mxz;
  real Myx, Myy, Myz;
  real Mzx, Mzy, Mzz;

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
          rx = x[ioffset    ] - x[joffset    ];
          ry = x[ioffset + 1] - x[joffset + 1];
          rz = x[ioffset + 2] - x[joffset + 2];

	  // Project a vector r to the extended unit cell
	  // centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If
	  // any dimension of L is equal or smaller than zero the
	  // box is assumed to be infinite in that direction.
	  if(Lx > 0){
	    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
            rz = rz + boxZ * Lz;
	  }

	  // 1. Compute mobility for pair i-j, if i==j use self-interation
          int j_image = j;
          if(boxX!=0 or boxY!=0 or boxZ!=0){
            j_image = -1;
          }

          // 1. Compute mobility for pair i-j
          mobilityWFRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
          Myx = -Mxy;
          Mzx = -Mxz;
          Mzy = -Myz;
          mobilityWFSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, i,j_image, invaGPU, x[joffset+2]/a);

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
  real pi = real(4.0) * atan(real(1.0));
  real norm_fact_t = 8 * pi * eta * a2;
  u[ioffset    ] = Ux / norm_fact_t;
  u[ioffset + 1] = Uy / norm_fact_t;
  u[ioffset + 2] = Uz / norm_fact_t;

  return;
}


__global__ void rotation_from_force_no_wall(const real *x,
					   const real *f,
					   real *u,
					   int number_of_blobs,
					   real eta,
					   real a,
                                           real Lx,
                                           real Ly,
                                           real Lz){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;

  real invaGPU = real(1.0) / a;

  real a2 = a*a;

  real Ux=0;
  real Uy=0;
  real Uz=0;

  real rx, ry, rz;

  real Mxx, Mxy, Mxz;
  real Myx, Myy, Myz;
  real Mzx, Mzy, Mzz;

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
          rx = x[ioffset    ] - x[joffset    ];
          ry = x[ioffset + 1] - x[joffset + 1];
          rz = x[ioffset + 2] - x[joffset + 2];

	  // Project a vector r to the extended unit cell
	  // centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If
	  // any dimension of L is equal or smaller than zero the
	  // box is assumed to be infinite in that direction.
	  if(Lx > 0){
	    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
            rz = rz + boxZ * Lz;
	  }

	  // 1. Compute mobility for pair i-j, if i==j use self-interation
          int j_image = j;
          if(boxX!=0 or boxY!=0 or boxZ!=0){
            j_image = -1;
          }

          // 1. Compute mobility for pair i-j
          mobilityWFRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
          Myx = -Mxy;
          Mzx = -Mxz;
          Mzy = -Myz;

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
  real pi = real(4.0) * atan(real(1.0));
  real norm_fact_t = 8 * pi * eta * a2;
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
__device__ void mobilityUTRPY(real rx,
			      real ry,
			      real rz,
			      real &Mxx,
			      real &Mxy,
			      real &Mxz,
			      real &Myy,
			      real &Myz,
			      real &Mzz,
			      int i,
			      int j,
                              real invaGPU){

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
    real r2 = rx*rx + ry*ry + rz*rz;
    real r = sqrt(r2);
    real r3 = r2*r;
    // We should not divide by zero but std::numeric_limits<real>::min() does not work in the GPU
    // real invr = (r > std::numeric_limits<real>::min()) ? (1.0 / r) : (1.0 / std::numeric_limits<real>::min())
    real invr3 = real(1.0) / r3;
    real c1;
    if(r>=2){
      Mxx =  0;
      Mxy =  rz * invr3;
      Mxz = -ry * invr3;
      Myy =  0;
      Myz =  rx * invr3;
      Mzz =  0;

    }
    else{
      c1 = real(0.5) * (real(1.0) - real(0.375) * r); // 3/8 = 0.375
      Mxx =  0;
      Mxy =  c1 * rz;
      Mxz = -c1 * ry ;
      Myy =  0;
      Myz =  c1 * rx;
      Mzz =  0;
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

__device__ void mobilityUTSingleWallCorrection(real rx,
			                       real ry,
			                       real rz,
			                       real &Mxx,
                  			       real &Mxy,
			                       real &Mxz,
                                               real &Myx,
			                       real &Myy,
			                       real &Myz,
                                               real &Mzx,
                                               real &Mzy,
			                       int i,
			                       int j,
                                               real invaGPU,
                                               real hj){
  if(i == j){
    real invZi = real(1.0) / hj;
    real invZi4 = pow(invZi,4);
    Mxy -= - invZi4 * real(0.125); // 3/24 = 0.125
    Myx -=   invZi4 * real(0.125); // 3/24 = 0.125
  }
  else{
    real h_hat = hj / rz;
    real invR = rsqrt(rx*rx + ry*ry + rz*rz); // = 1 / r;
    real invR2 = invR * invR;
    real invR4 = invR2 * invR2;
    real ex = rx * invR;
    real ey = ry * invR;
    real ez = rz * invR;

    real fact1 =  invR2;
    real fact2 = (6*h_hat*ez*ez*invR2 + (1-10*ez*ez)*invR4) * real(2.0);
    real fact3 = -ez*(3*h_hat*invR2 - 5*invR4) * real(2.0);
    real fact4 = -ez*(h_hat*invR2 - invR4) * real(2.0);

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


__global__ void velocity_from_force_and_torque(const real *x,
					       const real *f,
					       const real *t,
					       real *u,
					       int number_of_blobs,
					       real eta,
					       real a,
                                               real Lx,
                                               real Ly,
                                               real Lz){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;

  real invaGPU = real(1.0) / a;

  real a2 = a*a;

  real Ufx=0;
  real Ufy=0;
  real Ufz=0;

  real Utx=0;
  real Uty=0;
  real Utz=0;

  real rx, ry, rz;

  real Mxx, Mxy, Mxz;
  real Myx, Myy, Myz;
  real Mzx, Mzy, Mzz;

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
          rx = x[ioffset    ] - x[joffset    ];
          ry = x[ioffset + 1] - x[joffset + 1];
          rz = x[ioffset + 2] - x[joffset + 2];

	  // Project a vector r to the extended unit cell
	  // centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If
	  // any dimension of L is equal or smaller than zero the
	  // box is assumed to be infinite in that direction.
	  if(Lx > 0){
	    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
            rz = rz + boxZ * Lz;
	  }

	  // 1. Compute mobility for pair i-j, if i==j use self-interation
          int j_image = j;
          if(boxX!=0 or boxY!=0 or boxZ!=0){
            j_image = -1;
          }

          // 1. Compute UT mobility for pair i-j
          mobilityUTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
          Myx = -Mxy;
          Mzx = -Mxz;
          Mzy = -Myz;

          // Mind the correct symmety! M_UT,ij^{alpha,beta} = M_WF,ji^{beta,alpha}
          // mobilityUTSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, i,j, invaGPU, x[joffset+2]/a);
          mobilityUTSingleWallCorrection(-rx/a, -ry/a, (-rz+2*x[ioffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, j_image,i, invaGPU, x[ioffset+2]/a);

          // 2. Compute product M_ij * T_j
          Utx = Utx + (Mxx * t[joffset] + Mxy * t[joffset + 1] + Mxz * t[joffset + 2]);
          Uty = Uty + (Myx * t[joffset] + Myy * t[joffset + 1] + Myz * t[joffset + 2]);
          Utz = Utz + (Mzx * t[joffset] + Mzy * t[joffset + 1] + Mzz * t[joffset + 2]);

          // 3. Compute UF mobility for pair i-j
          mobilityUFRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
          Myx = Mxy;
          Mzx = Mxz;
          Mzy = Myz;
          mobilityUFSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy,Mzz, i,j_image, invaGPU, x[joffset+2]/a);

          // 4. Compute product M_ij * F_j
          Ufx = Ufx + (Mxx * f[joffset] + Mxy * f[joffset + 1] + Mxz * f[joffset + 2]);
          Ufy = Ufy + (Myx * f[joffset] + Myy * f[joffset + 1] + Myz * f[joffset + 2]);
          Ufz = Ufz + (Mzx * f[joffset] + Mzy * f[joffset + 1] + Mzz * f[joffset + 2]);
        }
      }
    }
  }
  //LOOP END

  //3. Save velocity U_i
  real pi = real(4.0) * atan(real(1.0));
  real norm_fact_t = 8 * pi * eta * a2;
  real norm_fact_f = 8 * pi * eta * a;
  u[ioffset    ] = Utx / norm_fact_t + Ufx / norm_fact_f;
  u[ioffset + 1] = Uty / norm_fact_t + Ufy / norm_fact_f;
  u[ioffset + 2] = Utz / norm_fact_t + Ufz / norm_fact_f;

  return;
}


__global__ void velocity_from_force_and_torque_no_wall(const real *x,
						       const real *f,
						       const real *t,
						       real *u,
						       int number_of_blobs,
						       real eta,
						       real a,
                                                       real Lx,
                                                       real Ly,
                                                       real Lz){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;

  real invaGPU = real(1.0) / a;

  real a2 = a*a;

  real Ufx=0;
  real Ufy=0;
  real Ufz=0;

  real Utx=0;
  real Uty=0;
  real Utz=0;

  real rx, ry, rz;

  real Mxx, Mxy, Mxz;
  real Myx, Myy, Myz;
  real Mzx, Mzy, Mzz;

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
          rx = x[ioffset    ] - x[joffset    ];
          ry = x[ioffset + 1] - x[joffset + 1];
          rz = x[ioffset + 2] - x[joffset + 2];

	  // Project a vector r to the extended unit cell
	  // centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If
	  // any dimension of L is equal or smaller than zero the
	  // box is assumed to be infinite in that direction.
	  if(Lx > 0){
	    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
            rz = rz + boxZ * Lz;
	  }

	  // 1. Compute mobility for pair i-j, if i==j use self-interation
          int j_image = j;
          if(boxX!=0 or boxY!=0 or boxZ!=0){
            j_image = -1;
          }

          // 1. Compute UT mobility for pair i-j
          mobilityUTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
          Myx = -Mxy;
          Mzx = -Mxz;
          Mzy = -Myz;

          // 2. Compute product M_ij * T_j
          Utx = Utx + (Mxx * t[joffset] + Mxy * t[joffset + 1] + Mxz * t[joffset + 2]);
          Uty = Uty + (Myx * t[joffset] + Myy * t[joffset + 1] + Myz * t[joffset + 2]);
          Utz = Utz + (Mzx * t[joffset] + Mzy * t[joffset + 1] + Mzz * t[joffset + 2]);

          // 3. Compute UF mobility for pair i-j
          mobilityUFRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
          Myx = Mxy;
          Mzx = Mxz;
          Mzy = Myz;

          // 4. Compute product M_ij * F_j
          Ufx = Ufx + (Mxx * f[joffset] + Mxy * f[joffset + 1] + Mxz * f[joffset + 2]);
          Ufy = Ufy + (Myx * f[joffset] + Myy * f[joffset + 1] + Myz * f[joffset + 2]);
          Ufz = Ufz + (Mzx * f[joffset] + Mzy * f[joffset + 1] + Mzz * f[joffset + 2]);
        }
      }
    }
  }
  //LOOP END

  //3. Save velocity U_i
  real pi = real(4.0) * atan(real(1.0));
  real norm_fact_t = 8 * pi * eta * a2;
  real norm_fact_f = 8 * pi * eta * a;
  u[ioffset    ] = Utx / norm_fact_t + Ufx / norm_fact_f;
  u[ioffset + 1] = Uty / norm_fact_t + Ufy / norm_fact_f;
  u[ioffset + 2] = Utz / norm_fact_t + Ufz / norm_fact_f;

  return;
}


__global__ void velocity_from_torque(const real *x,
 			             const real *t,
				     real *u,
				     int number_of_blobs,
				     real eta,
				     real a,
                                     real Lx,
                                     real Ly,
                                     real Lz){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;

  real invaGPU = real(1.0) / a;

  real a2 = a*a;

  real Utx=0;
  real Uty=0;
  real Utz=0;

  real rx, ry, rz;

  real Mxx, Mxy, Mxz;
  real Myx, Myy, Myz;
  real Mzx, Mzy, Mzz;

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
          rx = x[ioffset    ] - x[joffset    ];
          ry = x[ioffset + 1] - x[joffset + 1];
          rz = x[ioffset + 2] - x[joffset + 2];

	  // Project a vector r to the extended unit cell
	  // centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If
	  // any dimension of L is equal or smaller than zero the
	  // box is assumed to be infinite in that direction.
	  if(Lx > 0){
	    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
            rz = rz + boxZ * Lz;
	  }

	  // 1. Compute mobility for pair i-j, if i==j use self-interation
          int j_image = j;
          if(boxX!=0 or boxY!=0 or boxZ!=0){
            j_image = -1;
          }

          // 1. Compute UT mobility for pair i-j
          mobilityUTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
          Myx = -Mxy;
          Mzx = -Mxz;
          Mzy = -Myz;
          // Mind the correct symmety! M_UT,ij^{alpha,beta} = M_WF,ji^{beta,alpha}
          // mobilityUTSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, i,j, invaGPU, x[joffset+2]/a);
          mobilityUTSingleWallCorrection(-rx/a, -ry/a, (-rz+2*x[ioffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, j_image,i, invaGPU, x[ioffset+2]/a);

          // 2. Compute product M_ij * T_j
          Utx = Utx + (Mxx * t[joffset] + Mxy * t[joffset + 1] + Mxz * t[joffset + 2]);
          Uty = Uty + (Myx * t[joffset] + Myy * t[joffset + 1] + Myz * t[joffset + 2]);
          Utz = Utz + (Mzx * t[joffset] + Mzy * t[joffset + 1] + Mzz * t[joffset + 2]);
        }
      }
    }
  }
  //LOOP END

  //3. Save velocity U_i
  real pi = real(4.0) * atan(real(1.0));
  real norm_fact_t = real(1.0) / (8 * pi * eta * a2);
  u[ioffset    ] = Utx * norm_fact_t ;
  u[ioffset + 1] = Uty * norm_fact_t ;
  u[ioffset + 2] = Utz * norm_fact_t ;

  return;
}


__global__ void velocity_from_torque_in_plane(const real *x,
 			             const real *t,
				     real *u,
				     int number_of_blobs,
				     real eta,
				     real a,
                                     real Lx,
                                     real Ly,
                                     real Lz){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;

  real invaGPU = real(1.0) / a;

  real a2 = a*a;

  real Utx=0;
  real Uty=0;

  real rx, ry, rz;

  real Mxx, Mxy, Mxz;
  real Myx, Myy, Myz;
  real Mzx, Mzy, Mzz;

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
          rx = x[ioffset    ] - x[joffset    ];
          ry = x[ioffset + 1] - x[joffset + 1];
          rz = x[ioffset + 2] - x[joffset + 2];

	  // Project a vector r to the extended unit cell
	  // centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If
	  // any dimension of L is equal or smaller than zero the
	  // box is assumed to be infinite in that direction.
	  if(Lx > 0){
	    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
            rz = rz + boxZ * Lz;
	  }

	  // 1. Compute mobility for pair i-j, if i==j use self-interation
          int j_image = j;
          if(boxX!=0 or boxY!=0 or boxZ!=0){
            j_image = -1;
          }

          // 1. Compute UT mobility for pair i-j
          mobilityUTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
          Myx = -Mxy;
          Mzx = -Mxz;
          Mzy = -Myz;
          // Mind the correct symmety! M_UT,ij^{alpha,beta} = M_WF,ji^{beta,alpha}
          // mobilityUTSingleWallCorrection(rx/a, ry/a, (rz+2*x[joffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, i,j, invaGPU, x[joffset+2]/a);
          mobilityUTSingleWallCorrection(-rx/a, -ry/a, (-rz+2*x[ioffset+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy, j_image,i, invaGPU, x[ioffset+2]/a);

          // 2. Compute product M_ij * T_j
          Utx = Utx + (Mxx * t[joffset] + Mxy * t[joffset + 1]); // + Mxz * t[joffset + 2]
          Uty = Uty + (Myx * t[joffset] + Myy * t[joffset + 1]); // + Myz * t[joffset + 2]
        }
      }
    }
  }
  //LOOP END

  //3. Save velocity U_i
  real pi = real(4.0) * atan(real(1.0));
  real norm_fact_t = real(1.0) / (8 * pi * eta * a2);
  u[ioffset    ] = Utx * norm_fact_t ;
  u[ioffset + 1] = Uty * norm_fact_t ;
  u[ioffset + 2] = 0 ;

  return;
}



__global__ void velocity_from_torque_no_wall(const real *x,
                                             const real *t,
                                             real *u,
					     int number_of_blobs,
					     real eta,
					     real a,
                                             real Lx,
                                             real Ly,
                                             real Lz){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;

  real invaGPU = real(1.0) / a;
  real a2 = a*a;

  real Utx=0;
  real Uty=0;
  real Utz=0;

  real rx, ry, rz;

  real Mxx, Mxy, Mxz;
  real Myx, Myy, Myz;
  real Mzx, Mzy, Mzz;

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
          rx = x[ioffset    ] - x[joffset    ];
          ry = x[ioffset + 1] - x[joffset + 1];
          rz = x[ioffset + 2] - x[joffset + 2];

	  // Project a vector r to the extended unit cell
	  // centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If
	  // any dimension of L is equal or smaller than zero the
	  // box is assumed to be infinite in that direction.
	  if(Lx > 0){
	    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
            rz = rz + boxZ * Lz;
	  }

	  // 1. Compute mobility for pair i-j, if i==j use self-interation
          int j_image = j;
          if(boxX!=0 or boxY!=0 or boxZ!=0){
            j_image = -1;
          }

          // 1. Compute UT mobility for pair i-j
          mobilityUTRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j_image, invaGPU);
          Myx = -Mxy;
          Mzx = -Mxz;
          Mzy = -Myz;

          // 2. Compute product M_ij * T_j
          Utx = Utx + (Mxx * t[joffset] + Mxy * t[joffset + 1] + Mxz * t[joffset + 2]);
          Uty = Uty + (Myx * t[joffset] + Myy * t[joffset + 1] + Myz * t[joffset + 2]);
          Utz = Utz + (Mzx * t[joffset] + Mzy * t[joffset + 1] + Mzz * t[joffset + 2]);
        }
      }
    }
  }
  //LOOP END

  //3. Save velocity U_i
  real pi = real(4.0) * atan(real(1.0));
  real norm_fact_t = 8 * pi * eta * a2;
  u[ioffset    ] = Utx / norm_fact_t;
  u[ioffset + 1] = Uty / norm_fact_t;
  u[ioffset + 2] = Utz / norm_fact_t;

  return;
}


/*
 mobilityUFRPY computes the 3x3 RPY mobility
 between blobs i and j normalized with 8 pi eta a
*/
__device__ void mobilityUFSourceTarget(real rx,
			               real ry,
			               real rz,
			               real &Mxx,
			               real &Mxy,
			               real &Mxz,
                                       real &Myy,
			               real &Myz,
                                       real &Mzz,
                                       real a, /* radius_source */
                                       real b /* raduis_target*/){

  real fourOverThree = real(4.0) / real(3.0);
  real r2 = rx*rx + ry*ry + rz*rz;
  real r = sqrt(r2);

  real C1, C2;
  if(r > (b+a)){
    real a2 = a * a;
    real b2 = b * b;
    C1 = (1 + (b2+a2) / (3 * r2)) / r;
    C2 = ((1 - (b2+a2) / r2) / r2) / r;
  }
  else if(r > fabs(b-a)){
    real r3 = r2 * r;
    C1 = ((16*(b+a)*r3 - pow(pow(b-a,2) + 3*r2,2)) / (32*r3)) * fourOverThree / (b * a);
    C2 = ((3*pow(pow(b-a,2)-r2, 2) / (32*r3)) / r2) * fourOverThree / (b * a);
  }
  else{
    real largest_radius = (a > b) ? a : b;
    C1 = fourOverThree / (largest_radius);
    C2 = 0;
  }

  Mxx = C1 + C2 * rx * rx;
  Mxy =      C2 * rx * ry;
  Mxz =      C2 * rx * rz;
  Myy = C1 + C2 * ry * ry;
  Myz =      C2 * ry * rz;
  Mzz = C1 + C2 * rz * rz;

  return;
}


/*
  Wall corrections
*/
__device__ void mobilityUFSingleWallCorrectionSourceTarget(real rx,
                                                           real ry,
                                                           real rz,
                                                           real &Mxx,
                                                           real &Mxy,
                                                           real &Mxz,
                                                           real &Myx,
                                                           real &Myy,
                                                           real &Myz,
                                                           real &Mzx,
                                                           real &Mzy,
                                                           real &Mzz,
                                                           real a /*radius_source*/,
                                                           real b /*radius_target*/,
                                                           real y3,
                                                           real x3){
  real a2 = a * a;
  real b2 = b * b;
  real r2 = rx*rx + ry*ry + rz*rz;
  real r = sqrt(r2);
  real r3 = r2 * r;
  real r5 = r3 * r2;
  real r7 = r5 * r2;
  real r9 = r7 * r2;

  Mxx -= ((1+(b2+a2)/(real(3.0)*r2)) + (1-(b2+a2)/r2) * rx * rx / r2) / r;
  Mxy -= (                       (1-(b2+a2)/r2) * rx * ry / r2) / r;
  Mxz += (                       (1-(b2+a2)/r2) * rx * rz / r2) / r;
  Myx -= (                       (1-(b2+a2)/r2) * ry * rx / r2) / r;
  Myy -= ((1+(b2+a2)/(real(3.0)*r2)) + (1-(b2+a2)/r2) * ry * ry / r2) / r;
  Myz += (                       (1-(b2+a2)/r2) * ry * rz / r2) / r;
  Mzx -= (                       (1-(b2+a2)/r2) * rz * rx / r2) / r;
  Mzy -= (                       (1-(b2+a2)/r2) * rz * ry / r2) / r;
  Mzz += ((1+(b2+a2)/(real(3.0)*r2)) + (1-(b2+a2)/r2) * rz * rz / r2) / r;

  // M[l][m] += 2*(-J[l][m]/r - r[l]*x3[m]/r3 - y3[l]*r[m]/r3 + x3*y3*(I[l][m]/r3 - 3*r[l]*r[m]/r5));
  Mxx -= 2*(x3*y3*(real(1.0)/r3 - 3*rx*rx/r5));
  Mxy -= 2*(x3*y3*(       - 3*rx*ry/r5));
  Mxz += 2*(-rx*x3/r3 + x3*y3*( -3*rx*rz/r5));
  Myx -= 2*(x3*y3*(       - 3*ry*rx/r5));
  Myy -= 2*(x3*y3*(real(1.0)/r3 - 3*ry*ry/r5));
  Myz += 2*(-ry*x3/r3 + x3*y3*( -3*ry*rz/r5));
  Mzx -= 2*(-y3*rx/r3 + x3*y3*( -3*rz*rx/r5));
  Mzy -= 2*(-y3*ry/r3 + x3*y3*( -3*rz*ry/r5));
  Mzz += 2*(-real(1.0)/r - rz*x3/r3 - y3*rz/r3 + x3*y3*(real(1.0)/r3 - 3*rz*rz/r5));

  // M[l][m] += (2*b2/real(3.0)) * (-J[l][m]/r3 + 3*r[l]*rz[m]/r5 - y3*(3*rz*I[l][m]/r5 + 3*delta_3[l]*r[m]/r5 + 3*r[l]*delta_3[m]/r5 - 15*rz*r[l]*r[m]/r7));
  Mxx -= (2*b2/real(3.0)) * (-y3*(3*rz/r5 - 15*rz*rx*rx/r7));
  Mxy -= (2*b2/real(3.0)) * (-y3*(        - 15*rz*rx*ry/r7));
  Mxz += (2*b2/real(3.0)) * (3*rx*rz/r5 - y3*(3*rx/r5 - 15*rz*rx*rz/r7));
  Myx -= (2*b2/real(3.0)) * (-y3*(        - 15*rz*ry*rx/r7));
  Myy -= (2*b2/real(3.0)) * (-y3*(3*rz/r5 - 15*rz*ry*ry/r7));
  Myz += (2*b2/real(3.0)) * (3*ry*rz/r5 - y3*(3*ry/r5 - 15*rz*ry*rz/r7));
  Mzx -= (2*b2/real(3.0)) * (-y3*(3*rx/r5 - 15*rz*rz*rx/r7));
  Mzy -= (2*b2/real(3.0)) * (-y3*(3*ry/r5 - 15*rz*rz*ry/r7));
  Mzz += (2*b2/real(3.0)) * (-real(1.0)/r3 + 3*rz*rz/r5 - y3*(3*rz/r5 + 3*rz/r5 + 3*rz/r5 - 15*rz*rz*rz/r7));

  // M[l][m] += (2*a2/real(3.0)) * (-J[l][m]/r3 + 3*rz[l]*r[m]/r5 - x3*(3*rz*I[l][m]/r5 + 3*delta_3[l]*r[m]/r5 + 3*r[l]*delta_3[m]/r5 - 15*rz*r[l]*r[m]/r7));
  Mxx -= (2*a2/real(3.0)) * (-x3*(3*rz/r5 - 15*rz*rx*rx/r7));
  Mxy -= (2*a2/real(3.0)) * (-x3*(        - 15*rz*rx*ry/r7));
  Mxz += (2*a2/real(3.0)) * (-x3*(3*rx/r5 - 15*rz*rx*rz/r7));
  Myx -= (2*a2/real(3.0)) * (-x3*(        - 15*rz*ry*rx/r7));
  Myy -= (2*a2/real(3.0)) * (-x3*(3*rz/r5 - 15*rz*ry*ry/r7));
  Myz += (2*a2/real(3.0)) * (-x3*(3*ry/r5 - 15*rz*ry*rz/r7));
  Mzx -= (2*a2/real(3.0)) * (3*rz*rx/r5 - x3*(3*rx/r5 - 15*rz*rz*rx/r7));
  Mzy -= (2*a2/real(3.0)) * (3*rz*ry/r5 - x3*(3*ry/r5 - 15*rz*rz*ry/r7));
  Mzz += (2*a2/real(3.0)) * (-real(1.0)/r3 + 3*rz*rz/r5 - x3*(3*rz/r5 + 3*rz/r5 + 3*rz/r5 - 15*rz*rz*rz/r7));

  // M[l][m] += (2*b2*a2/real(3.0)) * (-I[l][m]/r5 + 5*rz*rz*I[l][m]/r7 - J[l][m]/r5 + 5*rz[l]*r[m]/r7 - J[l][m]/r5 + 5*r[l]*rz[m]/r7 + 5*rz[l]*r[m]/r7 + 5*r[l]*r[m]/r7 + 5*r[l]*rz[m]/r7 - 35 * rz*rz*r[l]*r[m]/r9);
  Mxx -= (2*b2*a2/real(3.0)) * (-real(1.0)/r5 + 5*rz*rz/r7 + 5*rx*rx/r7 - 35 * rz*rz*rx*rx/r9);
  Mxy -= (2*b2*a2/real(3.0)) * (          5*rx*ry/r7 +            - 35 * rz*rz*rx*ry/r9);
  Mxz += (2*b2*a2/real(3.0)) * (5*rx*rz/r7 + 5*rx*rz/r7 + 5*rx*rz/r7 - 35 * rz*rz*rx*rz/r9);
  Myx -= (2*b2*a2/real(3.0)) * (5*ry*rx/r7 - 35 * rz*rz*ry*rx/r9);
  Myy -= (2*b2*a2/real(3.0)) * (-real(1.0)/r5 + 5*rz*rz/r7 + 5*ry*ry/r7 - 35 * rz*rz*ry*ry/r9);
  Myz += (2*b2*a2/real(3.0)) * (5*ry*rz/r7 + 5*ry*rz/r7 + 5*ry*rz/r7 - 35 * rz*rz*rz*ry/r9);
  Mzx -= (2*b2*a2/real(3.0)) * (5*rz*rx/r7 + 5*rz*rx/r7 + 5*rz*rx/r7 - 35 * rz*rz*rz*rx/r9);
  Mzy -= (2*b2*a2/real(3.0)) * (5*rz*ry/r7 + 5*rz*ry/r7 + 5*rz*ry/r7 - 35 * rz*rz*rz*ry/r9);
  Mzz += (2*b2*a2/real(3.0)) * (-real(1.0)/r5 + 5*rz*rz/r7 - real(1.0)/r5 + 5*rz*rz/r7 - real(1.0)/r5 + 5*rz*rz/r7 + 5*rz*rz/r7 + 5*rz*rz/r7 + 5*rz*rz/r7 - 35 * rz*rz*rz*rz/r9);
}


/*
 velocity_from_force computes the product
 U = M*F
*/
__global__ void velocity_from_force_source_target(const real *y,
                                                  const real *x,
                                                  const real *f,
                                                  real *u,
                                                  const real *radius_source,
                                                  const real *radius_target,
				                  const int number_of_sources,
                                                  const int number_of_targets,
                                                  const real eta,
                                                  const real Lx,
                                                  const real Ly,
                                                  const real Lz){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_targets) return;

  real Ux=0;
  real Uy=0;
  real Uz=0;

  real rx, ry, rz;

  real Mxx, Mxy, Mxz;
  real Myx, Myy, Myz;
  real Mzx, Mzy, Mzz;

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
	for(int j=0; j<number_of_sources; j++){
	  joffset = j * NDIM;

	  // Compute vector between particles i and j
	  rx = x[ioffset    ] - y[joffset    ];
	  ry = x[ioffset + 1] - y[joffset + 1];
	  rz = x[ioffset + 2] - y[joffset + 2];

	  // Project a vector r to the extended unit cell
	  // centered around (0,0,0) and of size L=3*(Lx, Ly, Lz). If
	  // any dimension of L is equal or smaller than zero the
	  // box is assumed to be infinite in that direction.
	  if(Lx > 0){
	    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
            rz = rz + boxZ * Lz;
	  }

	  // 1. Compute mobility for pair i-j (unbounded contribution)
	  mobilityUFSourceTarget(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, radius_source[j], radius_target[i]);
          Myx = Mxy;
          Mzx = Mxz;
          Mzy = Myz;

	  mobilityUFSingleWallCorrectionSourceTarget(rx, ry, (rz+2*y[joffset+2]), Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy,Mzz, radius_source[j], radius_target[i], y[joffset+2], x[ioffset+2]);

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
  real pi = real(4.0) * atan(real(1.0));
  real norm_fact_f = real(1.0) / (8 * pi * eta);
  u[ioffset    ] = Ux * norm_fact_f;
  u[ioffset + 1] = Uy * norm_fact_f;
  u[ioffset + 2] = Uz * norm_fact_f;

  return;
}


/*
 velocity_from_force computes the product
 U = M*F
 with a free surface at z=0.
*/
__global__ void free_surface_velocity_from_force(const real *x,
                                                 const real *f,					
                                                 real *u,
				                 int number_of_blobs,
                                                 real eta,
                                                 real a,
                                                 real Lx,
                                                 real Ly,
                                                 real Lz){


  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i >= number_of_blobs) return;   

  real invaGPU = real(1.0) / a;

  real Ux=0;
  real Uy=0;
  real Uz=0;

  real rx, ry, rz;

  real Mxx, Mxy, Mxz;
  real Myx, Myy, Myz;
  real Mzx, Mzy, Mzz;
  real Mxx_image, Mxy_image, Mxz_image;
  real Myx_image, Myy_image, Myz_image;
  real Mzx_image, Mzy_image, Mzz_image;



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
	    rx = rx - int(rx / Lx + real(0.5) * (int(rx>0) - int(rx<0))) * Lx;
            rx = rx + boxX * Lx;
	  }
	  if(Ly > 0){
	    ry = ry - int(ry / Ly + real(0.5) * (int(ry>0) - int(ry<0))) * Ly;
            ry = ry + boxY * Ly;
	  }
	  if(Lz > 0){
	    rz = rz - int(rz / Lz + real(0.5) * (int(rz>0) - int(rz<0))) * Lz;
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
	  mobilityUFRPY(rx,ry,(rz+2*x[joffset+2]), Mxx_image,Mxy_image,Mxz_image,Myy_image,Myz_image,Mzz_image, i, -1, invaGPU);
	  Myx_image = Mxy_image;
	  Mzx_image = Mxz_image;
	  Mzy_image = Myz_image;

          // Add mobilities
          Mxx = Mxx + Mxx_image;
          Mxy = Mxy + Mxy_image;
          Mxz = Mxz - Mxz_image;
          Myx = Myx + Myx_image;
          Myy = Myy + Myy_image;
          Myz = Myz - Myz_image;
          Mzx = Mzx + Mzx_image;
          Mzy = Mzy + Mzy_image;
          Mzz = Mzz - Mzz_image;
	  
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
  real pi = real(4.0) * atan(real(1.0));
  real norm_fact_f = real(1.0) / (8 * pi * eta * a);
  u[ioffset    ] = Ux * norm_fact_f;
  u[ioffset + 1] = Uy * norm_fact_f;
  u[ioffset + 2] = Uz * norm_fact_f;

  return;
}

""")

def real(x):
  if precision == 'single':
    return np.float32(x)
  else:
    return np.float64(x)


def set_number_of_threads_and_blocks(num_elements):
  '''
  This functions uses a heuristic method to determine
  the number of blocks and threads per block to be
  used in CUDA kernels.
  '''
  threads_per_block=512
  if((num_elements//threads_per_block) < 512):
    threads_per_block = 256
  if((num_elements//threads_per_block) < 256):
    threads_per_block = 128
  if((num_elements//threads_per_block) < 128):
    threads_per_block = 64
  if((num_elements//threads_per_block) < 128):
    threads_per_block = 32
  num_blocks = (num_elements-1)//threads_per_block + 1

  return (threads_per_block, int(num_blocks))


def single_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):

  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = real(np.reshape(r_vectors, number_of_blobs * 3))
  f = real(np.reshape(force, number_of_blobs * 3))

  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  f_gpu = cuda.mem_alloc(f.nbytes)
  u_gpu = cuda.mem_alloc(f.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(f_gpu, f)

  # Get mobility function
  mobility = mod.get_function("velocity_from_force")

  # Compute mobility force product
  mobility(x_gpu, f_gpu, u_gpu, number_of_blobs, real(eta), real(a), real(L[0]), real(L[1]), real(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1))

  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(f)
  cuda.memcpy_dtoh(u, u_gpu)
  return u


def in_plane_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):

  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = real(np.reshape(r_vectors, number_of_blobs * 3))
  f = real(np.reshape(force, number_of_blobs * 3))

  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  f_gpu = cuda.mem_alloc(f.nbytes)
  u_gpu = cuda.mem_alloc(f.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(f_gpu, f)

  # Get mobility function
  mobility = mod.get_function("velocity_from_force_in_plane")

  # Compute mobility force product
  mobility(x_gpu, f_gpu, u_gpu, number_of_blobs, real(eta), real(a), real(L[0]), real(L[1]), real(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1))

  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(f)
  cuda.memcpy_dtoh(u, u_gpu)
  return u


def single_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):

  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = real(np.reshape(r_vectors, number_of_blobs * 3))
  f = real(np.reshape(force, number_of_blobs * 3))

  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  f_gpu = cuda.mem_alloc(f.nbytes)
  u_gpu = cuda.mem_alloc(f.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(f_gpu, f)

  # Get mobility function
  mobility = mod.get_function("rotation_from_force")

  # Compute mobility force product
  mobility(x_gpu, f_gpu, u_gpu, number_of_blobs, real(eta), real(a), real(L[0]), real(L[1]), real(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1))

  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(f)
  cuda.memcpy_dtoh(u, u_gpu)

  return u


def no_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):

  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = real(np.reshape(r_vectors, number_of_blobs * 3))
  f = real(np.reshape(force, number_of_blobs * 3))

  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  f_gpu = cuda.mem_alloc(f.nbytes)
  u_gpu = cuda.mem_alloc(f.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(f_gpu, f)

  # Get mobility function
  mobility = mod.get_function("rotation_from_force_no_wall")

  # Compute mobility force product
  mobility(x_gpu, f_gpu, u_gpu, number_of_blobs, real(eta), real(a), real(L[0]), real(L[1]), real(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1))

  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(f)
  cuda.memcpy_dtoh(u, u_gpu)

  return u


def single_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs):

  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = real(np.reshape(r_vectors, number_of_blobs * 3))
  t = real(np.reshape(torque, number_of_blobs * 3))

  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  t_gpu = cuda.mem_alloc(t.nbytes)
  u_gpu = cuda.mem_alloc(t.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(t_gpu, t)

  # Get mobility function
  mobility = mod.get_function("rotation_from_torque")

  # Compute mobility force product
  mobility(x_gpu, t_gpu, u_gpu, number_of_blobs, real(eta), real(a), real(L[0]), real(L[1]), real(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1))

  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(t)
  cuda.memcpy_dtoh(u, u_gpu)

  return u


def no_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs):

  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = real(np.reshape(r_vectors, number_of_blobs * 3))
  t = real(np.reshape(torque, number_of_blobs * 3))

  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  t_gpu = cuda.mem_alloc(t.nbytes)
  u_gpu = cuda.mem_alloc(t.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(t_gpu, t)

  # Get mobility function
  mobility = mod.get_function("rotation_from_torque_no_wall")

  # Compute mobility force product
  mobility(x_gpu, t_gpu, u_gpu, number_of_blobs, real(eta), real(a), real(L[0]), real(L[1]), real(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1))

  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(t)
  cuda.memcpy_dtoh(u, u_gpu)

  return u


def single_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force, torque, eta, a, *args, **kwargs):

  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = real(np.reshape(r_vectors, number_of_blobs * 3))
  f = real(np.reshape(force, number_of_blobs * 3))
  t = real(np.reshape(torque, number_of_blobs * 3))

  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  t_gpu = cuda.mem_alloc(t.nbytes)
  f_gpu = cuda.mem_alloc(f.nbytes)
  u_gpu = cuda.mem_alloc(t.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(t_gpu, t)
  cuda.memcpy_htod(f_gpu, f)

  # Get mobility function
  mobility = mod.get_function("velocity_from_force_and_torque")

  # Compute mobility force product
  mobility(x_gpu, f_gpu, t_gpu, u_gpu, number_of_blobs, real(eta), real(a), real(L[0]), real(L[1]), real(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1))

  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(t)
  cuda.memcpy_dtoh(u, u_gpu)

  return u  
  

def no_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force, torque, eta, a, *args, **kwargs):

  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = real(np.reshape(r_vectors, number_of_blobs * 3))
  f = real(np.reshape(force, number_of_blobs * 3))
  t = real(np.reshape(torque, number_of_blobs * 3))

  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  t_gpu = cuda.mem_alloc(t.nbytes)
  f_gpu = cuda.mem_alloc(f.nbytes)
  u_gpu = cuda.mem_alloc(t.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(t_gpu, t)
  cuda.memcpy_htod(f_gpu, f)

  # Get mobility function
  mobility = mod.get_function("velocity_from_force_and_torque_no_wall")

  # Compute mobility force product
  mobility(x_gpu, f_gpu, t_gpu, u_gpu, number_of_blobs, real(eta), real(a), real(L[0]), real(L[1]), real(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1))

  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(t)
  cuda.memcpy_dtoh(u, u_gpu)

  return u


def no_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):

  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = real(np.reshape(r_vectors, number_of_blobs * 3))
  f = real(np.reshape(force, number_of_blobs * 3))

  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  f_gpu = cuda.mem_alloc(f.nbytes)
  u_gpu = cuda.mem_alloc(f.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(f_gpu, f)

  # Get mobility function
  mobility = mod.get_function("velocity_from_force_no_wall")

  # Compute mobility force product
  mobility(x_gpu, f_gpu, u_gpu, number_of_blobs, real(eta), real(a), real(L[0]), real(L[1]), real(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1))

  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(x)
  cuda.memcpy_dtoh(u, u_gpu)

  return u


def single_wall_mobility_trans_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs):

  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = real(np.reshape(r_vectors, number_of_blobs * 3))
  t = real(np.reshape(torque, number_of_blobs * 3))

  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  t_gpu = cuda.mem_alloc(t.nbytes)
  u_gpu = cuda.mem_alloc(t.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(t_gpu, t)

  # Get mobility function
  mobility = mod.get_function("velocity_from_torque")

  # Compute mobility force product
  mobility(x_gpu, t_gpu, u_gpu, number_of_blobs, real(eta), real(a), real(L[0]), real(L[1]), real(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1))

  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(t)
  cuda.memcpy_dtoh(u, u_gpu)

  return u

def in_plane_mobility_trans_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs):

  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = real(np.reshape(r_vectors, number_of_blobs * 3))
  t = real(np.reshape(torque, number_of_blobs * 3))

  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  t_gpu = cuda.mem_alloc(t.nbytes)
  u_gpu = cuda.mem_alloc(t.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(t_gpu, t)

  # Get mobility function
  mobility = mod.get_function("velocity_from_torque_in_plane")

  # Compute mobility force product
  mobility(x_gpu, t_gpu, u_gpu, number_of_blobs, real(eta), real(a), real(L[0]), real(L[1]), real(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1))

  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(t)
  cuda.memcpy_dtoh(u, u_gpu)

  return u




def no_wall_mobility_trans_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs):

  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = real(np.reshape(r_vectors, number_of_blobs * 3))
  t = real(np.reshape(torque, number_of_blobs * 3))

  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  t_gpu = cuda.mem_alloc(t.nbytes)
  u_gpu = cuda.mem_alloc(t.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(t_gpu, t)

  # Get mobility function
  mobility = mod.get_function("velocity_from_torque_no_wall")

  # Compute mobility force product
  mobility(x_gpu, t_gpu, u_gpu, number_of_blobs, real(eta), real(a), real(L[0]), real(L[1]), real(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1))

  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(t)
  cuda.memcpy_dtoh(u, u_gpu)

  return u


def single_wall_mobility_trans_times_force_source_target_pycuda(source, target, force, radius_source, radius_target, eta, *args, **kwargs):

  # Determine number of threads and blocks for the GPU
  number_of_sources = np.int32(source.size // 3)
  number_of_targets = np.int32(target.size // 3)
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_targets)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = real(np.reshape(target, target.size))
  y = real(np.reshape(source, source.size))

  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  y_gpu = cuda.mem_alloc(y.nbytes)
  radius_target_gpu = cuda.mem_alloc(real(radius_target).nbytes)
  radius_source_gpu = cuda.mem_alloc(real(radius_source).nbytes)
  f_gpu = cuda.mem_alloc(real(force).nbytes)
  u_gpu = cuda.mem_alloc(x.nbytes)

  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(y_gpu, y)
  cuda.memcpy_htod(radius_target_gpu, real(radius_target))
  cuda.memcpy_htod(radius_source_gpu, real(radius_source))
  cuda.memcpy_htod(f_gpu, real(force))

  # Get mobility function
  mobility = mod.get_function("velocity_from_force_source_target")

  # Compute mobility force product
  mobility(y_gpu, x_gpu, f_gpu, u_gpu, radius_source_gpu, radius_target_gpu, np.int32(number_of_sources), np.int32(number_of_targets), real(eta), real(L[0]), real(L[1]), real(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1))

  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(x)
  cuda.memcpy_dtoh(u, u_gpu)
  return np.reshape(np.float64(u), u.size)


def free_surface_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):
   
  # Determine number of threads and blocks for the GPU
  number_of_blobs = np.int32(len(r_vectors))
  threads_per_block, num_blocks = set_number_of_threads_and_blocks(number_of_blobs)

  # Get parameters from arguments
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Reshape arrays
  x = real(np.reshape(r_vectors, number_of_blobs * 3))
  f = real(np.reshape(force, number_of_blobs * 3))
  
  # Allocate GPU memory
  x_gpu = cuda.mem_alloc(x.nbytes)
  f_gpu = cuda.mem_alloc(f.nbytes)
  u_gpu = cuda.mem_alloc(f.nbytes)
  number_of_blobs_gpu = cuda.mem_alloc(number_of_blobs.nbytes)
    
  # Copy data to the GPU (host to device)
  cuda.memcpy_htod(x_gpu, x)
  cuda.memcpy_htod(f_gpu, f)
    
  # Get mobility function
  mobility = mod.get_function("free_surface_velocity_from_force")

  # Compute mobility force product
  mobility(x_gpu, f_gpu, u_gpu, number_of_blobs, real(eta), real(a), real(L[0]), real(L[1]), real(L[2]), block=(threads_per_block, 1, 1), grid=(num_blocks, 1)) 
    
  # Copy data from GPU to CPU (device to host)
  u = np.empty_like(f)
  cuda.memcpy_dtoh(u, u_gpu)
  return u

