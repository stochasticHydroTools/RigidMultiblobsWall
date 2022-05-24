//################################################################################
//############### Basic C++ - Python bindings ####################################
//################################################################################
//################################################################################
//################# Interfacing Eigen and Python without copying data ############
//################################################################################
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <cmath>
#include <omp.h>
#include<lapacke.h>
#include <chrono>
#include <random>
#include<vector>
#include<iostream>
#include<algorithm>

/*
#include"uammd_interface.h"
using PSE = UAMMD_PSE_Glue;
using PSEParameters = PyParameters;
*/

#include"libMobility/solvers/NBody/mobility.h"
#include"libMobility/solvers/NBody_wall/mobility.h"
#include"libMobility/solvers/PSE/mobility.h"

using real = libmobility::real;




// Double Typedefs
typedef Eigen::VectorXd Vector;
typedef Eigen::Ref<Vector> RefVector;
typedef Eigen::Ref<const Vector> CstRefVector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::Ref<Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> RefMatrix;

typedef Eigen::Triplet<real> Trip;
typedef Eigen::SparseMatrix<real> SparseM;

typedef Eigen::Triplet<double> Trip_d;
typedef Eigen::SparseMatrix<double> SparseMd;
// Float typedefs
/*
typedef Eigen::VectorXf Vector;
typedef Eigen::Ref<Vector> RefVector;
typedef Eigen::Ref<const Vector> CstRefVector;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::Ref<Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> RefMatrix;
*/
typedef Eigen::VectorXi IVector;
typedef Eigen::Ref<IVector> IRefVector;
typedef Eigen::Ref<const IVector> CstIRefVector;


struct SpecialParameter{
  real psi = -1.0;
  real Lx = -1, Ly=-1, Lz=-1;
  std::string algorithmForNBody = "default";
};

enum class geometry{rpy, triply_periodic, single_wall};
auto createSolver(geometry geom, SpecialParameter parStar){
  std::shared_ptr<libmobility::Mobility> solver;
  if(geom == geometry::triply_periodic){
    libmobility::Configuration conf; 
    conf.periodicity=libmobility::periodicity_mode::triply_periodic; 
    auto pse = std::make_shared<PSE>(conf);
    //pse->setParametersPSE(parStar.psi, {parStar.Lx, parStar.Ly, parStar.Lz});
    solver = pse;
  }
  else if(geom == geometry::rpy){
    libmobility::Configuration conf; 
    conf.periodicity=libmobility::periodicity_mode::open; 
    auto nb = std::make_shared<NBody>(conf);
    solver = nb;
  }
  else if(geom == geometry::single_wall){
    libmobility::Configuration conf; 
    conf.periodicity=libmobility::periodicity_mode::single_wall; 
    auto nbw = std::make_shared<NBody_wall>(conf);
    solver = nbw;
  }
  else{
  	throw std::runtime_error("This solver is not implemented");
  }
  return solver;
}

  /*
    mobilityUFRPY computes the 3x3 RPY mobility
    between blobs i and j normalized with 8 pi eta a
  */
  void mobilityUFRPY(real rx, real ry, real rz,
                   real &Mxx, real &Mxy, real &Mxz,
                   real &Myy, real &Myz, real &Mzz,
                   int i, int j,
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
      real r = std::sqrt(r2);
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
  void mobilityUFSingleWallCorrection(real rx, real ry, real rz,
                                      real &Mxx, real &Mxy, real &Mxz,
                                      real &Myx, real &Myy, real &Myz,
                                      real &Mzx, real &Mzy, real &Mzz,
                                      int i, int j,
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
      real invR = 1.0/std::sqrt(rx*rx + ry*ry + rz*rz); // = 1 / r; //TODO: Make this a fast inv sqrt
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




class CManyFibers{
  real a, ds, dt, k_bend, M0, kBT, eta, Lp;
  //PSEParameters par, par_Mhalf;
  std::shared_ptr<libmobility::Mobility> solver;
  SpecialParameter parStar;
  bool clamp; // fibers clamped at one end or not
  Vector T_fix; // Ghost vector for clamped fibers (all use this same 3x1 for now)
  // Solver parameters
  real impl;
  real alpha;// = 2.0*impl*M0;
  real scale_00; // = M0/(1.0+alpha);
  real scale_10; // = M0;
  real scale; // scale for impl_hydro block
  bool PC_wall = false; // use wall corrections in PC or not
  bool perm_set = false;
  IVector perm;
  IVector i_perm;
  
  // Fiber configurations
  Vector X_0_h, X_0_p, X_0_m;
  Matrix T_h, U_h, V_h;
  Matrix T_p, U_p, V_p;
  Matrix T_m, U_m, V_m;
  bool parametersSet = false;
  //tbb::global_control tbbControl;
public:
  // CFibers():
  //   tbbControl(tbb::global_control::max_allowed_parallelism, 3){     
  // }

  void setParametersPSE(real psi){
    parStar.psi = psi;
  }
    
  void setParameters(int numParts, real a, real ds, real dt, real k_bend, real M0, real impl, real kBT, real eta, real Lp, bool clamp, Vector& T_fix){
    // TODO: Put the list of parameters into a structure
    this->a = a;
    this->ds = ds;
    this->dt = dt;
    this->k_bend = k_bend;
    this->M0 = M0;
    this->kBT = kBT;
    this->eta = eta;
    this->Lp = Lp;
    this->clamp = clamp;
    this->T_fix = T_fix;
    this->parametersSet = true;
    // solver parameters
    this->impl = impl;
    this->alpha = 2.0*impl*M0;
    this->scale_00 = M0/(1.0+alpha);
    this->scale_10 = M0;
    real K_scale = 2.0*ds;
    this->scale = (M0/(1+alpha))/(K_scale);
    
    
    geometry geom = geometry::triply_periodic;
    if(geom == geometry::single_wall){
      PC_wall = true;
    }
    parStar.Lx = Lp; parStar.Ly = Lp; parStar.Lz = Lp;
    solver = createSolver(geom, parStar);
    libmobility::Parameters par;
    par.hydrodynamicRadius = {a};
    par.viscosity = eta;
    par.temperature = 0.5;
    par.numberParticles = numParts;
    //par.boxSize = {Lp, Lp, Lp};
    solver->initialize(par);
  }
  
  void update_T_fix(Vector& T_fix){
    this->T_fix = T_fix;
  }
  
  std::vector<real> single_fiber_Pos(Matrix& T, Vector& X_0){
    const int N_lk = T.cols();
    Eigen::Vector3d T_j, Sum;
    Sum[0] = X_0[0];
    Sum[1] = X_0[1];
    Sum[2] = X_0[2];
    
    int size = 3*N_lk + 3;
    std::vector<real> pos;
    pos.reserve(size);
    
    pos.push_back(Sum[0]);
    pos.push_back(Sum[1]);
    pos.push_back(Sum[2]);
    
    // Rotate the frame
    for(int j = 0; j < N_lk; ++j){
        // Set components of the frame
        T_j = T.col(j);
        Sum += ds*T_j;
        pos.push_back(Sum[0]);
        pos.push_back(Sum[1]);
        pos.push_back(Sum[2]);
    }
    return pos;

  }
  
  
  template<class AMatrix, class BVector>
  std::vector<real> multi_fiber_Pos(AMatrix& T, BVector& X_0){
    const int N_lk = T.cols();
    const int N_fib = T.rows()/3;
    
    int size = N_fib*(3*N_lk + 3);
    std::vector<real> pos;
    std::vector<real> pos_j;
    pos.reserve(size);
    Matrix T_j(3,N_lk);
    Vector X_0_j(3);
    for(int j = 0; j < N_fib; ++j){
        for(int k = 0; k < N_lk; ++k){
            T_j(0,k) = T(3*j+0,k);
            T_j(1,k) = T(3*j+1,k);
            T_j(2,k) = T(3*j+2,k);
            //
            X_0_j(0) = X_0(3*j+0);
            X_0_j(1) = X_0(3*j+1);
            X_0_j(2) = X_0(3*j+2);
        }
        pos_j = single_fiber_Pos(T_j, X_0_j);
        pos.insert( pos.end(), pos_j.begin(), pos_j.end() );
    }
    return pos;

  }
  
  template<class AVector, class AMatrix>
  Matrix Kinv_multi(AVector& Vel, AMatrix& U, AMatrix& V){
    const int N_lk = U.cols();
    const int N_fib = U.rows()/3;
    
    Eigen::Vector3d U_j, V_j;
    Eigen::Vector3d Vel_j, Vel_jp1;
    Eigen::Vector3d Dp;
    
    int offset = 3;
    if(clamp){offset = 0;}
    
    Matrix Om = Matrix::Zero(N_fib,offset+2*N_lk);
    
    // set the first three elements of Om to Vel[0:3]
    int size = 3*(N_lk+1);
    
    if(!clamp){
        for(int k = 0; k < N_fib; ++k){
            for(int d = 0; d < 3; ++d){
                Om(k,d) = Vel(k*size+d);
            }
        }
    }
    
    for(int k = 0; k < N_fib; ++k){
        for(int j = 0; j < N_lk; ++j){
            // Set components of the frame
            for(int d = 0; d < 3; ++d){
                U_j(d) = U(3*k+d,j);
                V_j(d) = V(3*k+d,j);
            }
            
            for(int d = 0; d < 3; ++d){
                Dp[d] = (1/ds)*(Vel[k*size+3*j+3+d]-Vel[k*size+3*j+d]);
            }
            
            Om(k,offset+(2*j)) = -1.0*V_j.dot(Dp);
            Om(k,offset+(2*j+1)) = U_j.dot(Dp);
        }
    }
    
    
    return Om;
  }
  
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

///////////////////////////   PRECONDITIONER FUNCTIONS

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  template<class AMatrix, class BVector>
  SparseM Banded_rotne_prager_tensor(AMatrix& T, BVector& X_0, int bands) {
    
    std::vector<real> r_vectors = multi_fiber_Pos(T, X_0);

    // Compute scalar functions f(r) and g(r)
    real norm_fact_f = real(1.0) / (8.0 * M_PI * eta * a);

    // Build mobility matrix of size 3N \times 3N
    int N = r_vectors.size();
    int Nparts = N/3;
    
    real invaGPU = real(1.0) / a;
    
    real rx, ry, rz;

    real Mxx, Mxy, Mxz;
    real Myx, Myy, Myz;
    real Mzx, Mzy, Mzz;
    
    std::vector<Trip> tripletList;
    tripletList.reserve(Nparts*3*3*(2*bands+1));
    
    for (int i = 0; i < Nparts; ++i) {
        int lower_b = std::max(0,i-bands);
        int upper_b = std::min(Nparts-1,i+bands);
        for (int j = lower_b; j <= upper_b; ++j) {
            rx = r_vectors[3*i+0] - r_vectors[3*j+0];
            ry = r_vectors[3*i+1] - r_vectors[3*j+1];
            rz = r_vectors[3*i+2] - r_vectors[3*j+2];

            mobilityUFRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, invaGPU);
            Myx = Mxy;
            Mzx = Mxz;
            Mzy = Myz;
            if(PC_wall){
              mobilityUFSingleWallCorrection(rx/a, ry/a, (rz+2*r_vectors[3*j+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy,Mzz, i,j, r_vectors[3*j+2]/a);
            }
            
            tripletList.push_back(Trip(i * 3, j * 3,Mxx));
            tripletList.push_back(Trip(i * 3, j * 3 + 1,Mxy));
            tripletList.push_back(Trip(i * 3, j * 3 + 2,Mxz));

            tripletList.push_back(Trip(i * 3 + 1, j * 3,Myx));
            tripletList.push_back(Trip(i * 3 + 1, j * 3 + 1,Myy));
            tripletList.push_back(Trip(i * 3 + 1, j * 3 + 2,Myz));

            tripletList.push_back(Trip(i * 3 + 2, j * 3,Mzx));
            tripletList.push_back(Trip(i * 3 + 2, j * 3 + 1,Mzy));
            tripletList.push_back(Trip(i * 3 + 2, j * 3 + 2,Mzz));
        }
    }
    SparseM mat(N,N);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    
    mat *= norm_fact_f;
    
    return mat;
  }

  SparseM Banded_D4_v(int Nparts){
    const int Nm1 = Nparts-1;
    const int Nm2 = Nparts-2;
    
    std::vector<Trip> tripletList;
    tripletList.reserve(Nparts*3*(3*5));
    for (int j = 0; j < Nparts; ++j) {
        if(j==0){
            if(!clamp){
                for(int d = 0; d < 3; ++d){
                    tripletList.push_back(Trip(j*3+d, j*3+d, 1.0*impl));
                    tripletList.push_back(Trip(j*3+d, (j+1)*3+d, -2.0*impl));
                    tripletList.push_back(Trip(j*3+d, (j+2)*3+d, 1.0*impl));
                }
            }
        }
        else if(j==1){
            if(!clamp){
                for(int d = 0; d < 3; ++d){
                    tripletList.push_back(Trip(j*3+d, (j-1)*3+d, -2.0*impl));
                    tripletList.push_back(Trip(j*3+d, j*3+d, 5.0*impl));
                    tripletList.push_back(Trip(j*3+d, (j+1)*3+d, -4.0*impl));
                    tripletList.push_back(Trip(j*3+d, (j+2)*3+d, 1.0*impl));
                }
            }
            else {
                for(int d = 0; d < 3; ++d){
                    tripletList.push_back(Trip(j*3+d, (j-1)*3+d, -3.0*impl));
                    tripletList.push_back(Trip(j*3+d, j*3+d, 6.0*impl));
                    tripletList.push_back(Trip(j*3+d, (j+1)*3+d, -4.0*impl));
                    tripletList.push_back(Trip(j*3+d, (j+2)*3+d, 1.0*impl));
                } 
            }
        }
        else if(j==Nm2){
            for(int d = 0; d < 3; ++d){
                tripletList.push_back(Trip(j*3+d, (j+1)*3+d, -2.0*impl));
                tripletList.push_back(Trip(j*3+d, j*3+d, 5.0*impl));
                tripletList.push_back(Trip(j*3+d, (j-1)*3+d, -4.0*impl));
                tripletList.push_back(Trip(j*3+d, (j-2)*3+d, 1.0*impl));
            }
        }
        else if(j==Nm1){
            for(int d = 0; d < 3; ++d){
                tripletList.push_back(Trip(j*3+d, j*3+d, 1.0*impl));
                tripletList.push_back(Trip(j*3+d, (j-1)*3+d, -2.0*impl));
                tripletList.push_back(Trip(j*3+d, (j-2)*3+d, 1.0*impl));
            }
        }
        else{
            for(int d = 0; d < 3; ++d){
                tripletList.push_back(Trip(j*3+d, (j+2)*3+d, 1.0*impl));
                tripletList.push_back(Trip(j*3+d, (j+1)*3+d, -4.0*impl));
                tripletList.push_back(Trip(j*3+d, j*3+d, 6.0*impl));
                tripletList.push_back(Trip(j*3+d, (j-1)*3+d, -4.0*impl));
                tripletList.push_back(Trip(j*3+d, (j-2)*3+d, 1.0*impl));
            }
        }
    } // end i loop
    SparseM mat(3*Nparts,3*Nparts);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
  }

  template<class AMatrix>
  SparseM Sparse_B_mat(AMatrix& T){
    //
    //Return result of B T = d_s( T t_hat )
    //        
    int Nlk = T.cols();
    
    std::vector<Trip> tripletList;
    tripletList.reserve((Nlk+1)*3*(2));
    
    // End points
    int offset;
    if(!clamp){
        offset = 0;
        tripletList.push_back(Trip(0, 0, -T(0,0)));
        tripletList.push_back(Trip(1, 0, -T(1,0)));
        tripletList.push_back(Trip(2, 0, -T(2,0)));
    }
    else{
        offset = 3;
        tripletList.push_back(Trip(0, 0, 1.0));
        tripletList.push_back(Trip(1, 1, 1.0));
        tripletList.push_back(Trip(2, 2, 1.0));
    }
    //
    tripletList.push_back(Trip(3*Nlk+0, offset+Nlk-1, T(0,Nlk-1)));
    tripletList.push_back(Trip(3*Nlk+1, offset+Nlk-1, T(1,Nlk-1)));
    tripletList.push_back(Trip(3*Nlk+2, offset+Nlk-1, T(2,Nlk-1)));
    
    for (int j = 1; j < Nlk; ++j) {
        // cumsum of ds*tau_cross_Om
        tripletList.push_back(Trip(3*j+0, offset+j, -T(0,j)));
        tripletList.push_back(Trip(3*j+1, offset+j, -T(1,j)));
        tripletList.push_back(Trip(3*j+2, offset+j, -T(2,j)));
        //
        tripletList.push_back(Trip(3*j+0, offset+j-1, T(0,j-1)));
        tripletList.push_back(Trip(3*j+1, offset+j-1, T(1,j-1)));
        tripletList.push_back(Trip(3*j+2, offset+j-1, T(2,j-1)));
    }
    SparseM mat(3*Nlk+3,Nlk+offset);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
  }

  auto Banded_PC_Mat(Matrix& T, Vector& X_0, int bands) {
    int Nlk = T.cols();
    int Nparts = Nlk+1;
        
    std::vector<real> data;
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    
    // ###############################
    // ###############################
    // Assign the Big (0,0) block
    // I + impl*M_loc*D^4
    // ###############################
    // ###############################
    SparseM RPY = Banded_rotne_prager_tensor(T, X_0, bands);
    SparseM D4_v = Banded_D4_v(Nparts);
    SparseM I_MD4_v = (RPY * D4_v).pruned();
    for (int k = 0; k < I_MD4_v.rows(); ++k){
        I_MD4_v.coeffRef(k,k) += 1.0;
    }
    
    for (int k = 0; k < I_MD4_v.outerSize(); ++k){
        for (SparseM::InnerIterator it(I_MD4_v,k); it; ++it)
        {
            data.push_back(scale_00*it.value());
            row_idx.push_back(i_perm(it.row()));   // row index
            col_idx.push_back(i_perm(it.col()));   // col index (here it is equal to k)
        }
    }
    
    // ###############################
    // ###############################
    // Assign the Big (0,1) block
    // -M_loc*B
    // ###############################
    // ###############################
    const int OFFSET = 3*Nlk+3;
    SparseM B_mat = Sparse_B_mat(T);
    SparseM M_x_B = (RPY * B_mat).pruned();
    for (int k = 0; k < M_x_B.outerSize(); ++k){
        for (SparseM::InnerIterator it(M_x_B,k); it; ++it)
        {
            data.push_back(-1.0*it.value());
            row_idx.push_back(i_perm(it.row()));   // row index
            col_idx.push_back(i_perm(OFFSET+it.col()));   // col index (here it is equal to k)
        }
    }
    // ###############################
    // ###############################
    // Assign the Big (1,0) block
    // -B^{T}
    // ###############################
    // ###############################
    for (int k = 0; k < B_mat.outerSize(); ++k){
        for (SparseM::InnerIterator it(B_mat,k); it; ++it)
        {
            data.push_back(-scale_10*it.value());
            row_idx.push_back(i_perm(OFFSET+it.col()));   // row index
            col_idx.push_back(i_perm(it.row()));   // col index (here it is equal to k)
        }
    }

    return std::make_tuple(data,row_idx,col_idx);
  }

  std::tuple<Vector,Vector> Solve_Mband_Sys(Vector RHS, std::vector<real> data, std::vector<int> row_idx, std::vector<int> col_idx, Matrix& U, Matrix& V, int bands){ 
    // size variables
    int nz = perm.size();
    int N_blb = RHS.size()/3;
    
    
    // Mat the data structures for LAPACKs banded solver
    const int N = data.size();
    int b;
    Matrix AB = Matrix::Zero(2*bands+bands+1,nz);
    for(int i = 0; i < N; ++i){
        b = col_idx[i]-row_idx[i];
        AB(2*bands-b,col_idx[i]) = data[i];
    }
    
    // permute RHS
    int offset = 0;
    if(clamp){offset = 3;}
    
    Vector RHS_p = Vector::Zero(nz);
    Vector Z_pad = Vector::Zero(N_blb-1+offset);
    Vector RHS_pad(RHS.size() + Z_pad.size());
    RHS_pad << RHS, Z_pad;
    for(int k = 0; k < nz; ++k){
        RHS_p[k] = RHS_pad[perm[k]];
    }
        
    // Solve system
    // setup lapack parameters
    int n = nz;
    int kl = bands;
    int ku = bands;
    int ldab = 2*kl + ku +1;
    int nrhs = 1;
    int ipiv[nz];
    int ierr;
    
    // solve system
    dgbsv_(&n, &kl, &ku, &nrhs, & *AB.data(), &ldab, ipiv, & *RHS_p.data(), &n, &ierr);
    if(ierr != 0){std::cout << "lapack did bad" << std::endl;}
                
    // Extract solution
    Vector Vel = Vector::Zero(3*N_blb);
    Vector Tens = Vector::Zero(N_blb-1+offset);
    Matrix Om;
    
    if(clamp){
        Tens[0] = RHS_p[0];
        Tens[1] = RHS_p[1];
        Tens[2] = RHS_p[2];
    }
    
    for(int j = 0; j < N_blb; ++j){
        for(int d = 0; d < 3; ++d){
            Vel[3*j+d] = RHS_p[offset+4*j+d];
        }
        if(j!=(N_blb-1)){Tens[offset+j] = RHS_p[offset+4*j+3];}
    }
       
    Om = Kinv_multi(Vel, U, V);
    Om *= scale_00;

    Vector Om_v(Eigen::Map<Vector>(Om.data(), Om.cols()*Om.rows()));
    
    return std::make_tuple(Om_v,Tens);
  }
  
  
  std::tuple<Matrix,Matrix> apply_Banded_PC(Vector& X, RefMatrix& T, RefMatrix& U, RefMatrix& V, RefVector& X_0, int bands){
    //
    // apply the banded precoditioner
    //
    int Nlk = T.cols();
    int N_blb = Nlk+1;
    
    int offset = 0;
    if(clamp){offset = 3;}
    
    if(!perm_set){
        perm = IVector::Zero(4*N_blb-1+offset);
        i_perm = IVector::Zero(4*N_blb-1+offset);
        
        if(clamp){
            for(int d = 0; d < 3; ++d){
                perm[d] = 3*N_blb+d;
                i_perm[3*N_blb+d] = d;
            }
        }
        
        // make permutation vector
        for(int j = 0; j < N_blb; ++j){
            for(int d = 0; d < 3; ++d){
                perm[4*j+d+offset] = 3*j+d;
                i_perm[3*j+d] = 4*j+d+offset;
            }
            if(j!=(N_blb-1)){
                perm[4*j+3+offset] = 3*N_blb+j+offset;
                i_perm[3*N_blb+j+offset] = 4*j+3+offset;
            }
        }
        perm_set = true;
    }
    // Get matrix entries    
    std::vector<real> data;
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    
    // Get bands of Big matrix from M_bands
    int A_bands = 9;
    if(bands == 1){A_bands += 5;}
    if(bands == 2){A_bands += 10;}
    if(bands >= 3){A_bands += (10+(bands-2)*4);}
    
    // Make Om and Tans for all fibers
    const int N_fib = T.rows()/3;
    const int size_j = 3+3*Nlk;
    Matrix Om(N_fib,(3+2*Nlk-offset));
    Matrix Tens(N_fib,offset+Nlk);
    Vector Om_j, Tens_j;
    
    
    Matrix T_j(3,Nlk);
    Matrix U_j(3,Nlk);
    Matrix V_j(3,Nlk);
    Vector X_0_j(3);
    
    Vector RHS_j;
    
    for(int j = 0; j < N_fib; ++j){
        for(int k = 0; k < Nlk; ++k){
            T_j(0,k) = T(3*j+0,k);
            T_j(1,k) = T(3*j+1,k);
            T_j(2,k) = T(3*j+2,k);
            //
            U_j(0,k) = U(3*j+0,k);
            U_j(1,k) = U(3*j+1,k);
            U_j(2,k) = U(3*j+2,k);
            //
            V_j(0,k) = V(3*j+0,k);
            V_j(1,k) = V(3*j+1,k);
            V_j(2,k) = V(3*j+2,k);
            //
            X_0_j(0) = X_0(3*j+0);
            X_0_j(1) = X_0(3*j+1);
            X_0_j(2) = X_0(3*j+2);
        }
        std::tie(data,row_idx,col_idx) = Banded_PC_Mat(T_j, X_0_j, bands);
        
        RHS_j = X.segment(j*size_j,size_j);
        
        
        std::tie(Om_j,Tens_j) = Solve_Mband_Sys(RHS_j, data, row_idx, col_idx, U_j, V_j, A_bands);
        
        
        Om.row(j) = Om_j;
        Tens.row(j) = Tens_j;
        
        data.clear();
        row_idx.clear();
        col_idx.clear();
    }

    return std::make_tuple(Om, Tens);
    
  }
  
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

///////////////////////////   LINEAR OPERATOR FUNCTIONS

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  template<class AVector, class AMatrix, class BVector>
  Vector apply_M(AVector& F, AMatrix& T, BVector& X_0){
      
    // TODO:  make function "void updateSolverPositions(RefMatrix T, X_0){ " next 3 lines
    std::vector<real> pos = multi_fiber_Pos(T, X_0);
    int numberParticles = pos.size()/3;
    solver->setPositions(pos.data());
    
    //std::vector<real> MF(3*numberParticles, 0);
    Eigen::VectorXf MF(3*numberParticles);
    Eigen::VectorXf Ff = F.template cast <real> ();
    
    
    
    solver->Mdot(Ff.data(), nullptr, MF.data());
    Vector MFd = MF.cast <double> ();
    return MFd;
  }
  
  template<class AVector, class BVector>
  Vector apply_M_rvecs(AVector& F, BVector& pos){
      
    // TODO:  make function "void updateSolverPositions(RefMatrix T, X_0){ " next 3 lines
    int numberParticles = pos.size()/3;
    solver->setPositions(pos.data());
    
    //std::vector<real> MF(3*numberParticles, 0);
    Eigen::VectorXf MF(3*numberParticles);
    Eigen::VectorXf Ff = F.template cast <real> ();
    
    solver->Mdot(Ff.data(), nullptr, MF.data());
    Vector MFd = MF.cast <double> ();
    return MFd;
  }

  Vector apply_Mhalf_W(RefMatrix& T, RefVector& X_0){
    std::vector<real> pos = multi_fiber_Pos(T, X_0);
    int numberParticles = pos.size()/3;
    solver->setPositions(pos.data());
    
    Eigen::VectorXf Mhalf(3*numberParticles);
    
    real prefactor = 1.0;
    solver->stochasticDisplacements(Mhalf.data(), prefactor);
    Vector MhalfFd = Mhalf.cast <double> ();
    return MhalfFd;
  }
  
  Vector ds_D4_v(Vector& X) {
    //
    // Return D^4 * X
    //
    // Build mobility matrix of size 3N \times 3N
    int N = X.size();
    int Nblobs = N/3;
    const int Nm1 = Nblobs-1;
    const int Nm2 = Nblobs-2;
    Vector out = Vector::Zero(N);
        
    for (int j = 0; j < Nblobs; ++j) {
            if(j==0){
                if(!clamp){
                out(0) =  impl * (X(0) - 2.0 * X(3) + X(6));
                out(1) =  impl * (X(1) - 2.0 * X(4) + X(7));
                out(2) =  impl * (X(2) - 2.0 * X(5) + X(8));
                }
            }
            else if(j==1){
                real BC = static_cast<real>(clamp);
                out(3) = impl * ((-2.0-BC) * X(0) + (5.0+BC) * X(3) - 4.0 * X(6) + X(9));
                out(4) = impl * ((-2.0-BC) * X(1) + (5.0+BC) * X(4) - 4.0 * X(7) + X(10));
                out(5) = impl * ((-2.0-BC) * X(2) + (5.0+BC) * X(5) - 4.0 * X(8) + X(11));
            }
            else if(j==Nm2){
                out(3*j+0) = impl * (-2.0 * X(3*(j+1)) + 5.0 * X(3*j) - 4.0 * X(3*(j-1)) + X(3*(j-2)));
                out(3*j+1) = impl * (-2.0 * X(3*(j+1)+1) + 5.0 * X(3*j+1) - 4.0 * X(3*(j-1)+1) + X(3*(j-2)+1));
                out(3*j+2) = impl * (-2.0 * X(3*(j+1)+2) + 5.0 * X(3*j+2) - 4.0 * X(3*(j-1)+2) + X(3*(j-2)+2));
            }
            else if(j==Nm1){
                out(3*j+0) = impl * (X(3*j) - 2.0 * X(3*(j-1)) + X(3*(j-2)));
                out(3*j+1) = impl * (X(3*j+1) - 2.0 * X(3*(j-1)+1) + X(3*(j-2)+1));
                out(3*j+2) = impl * (X(3*j+2) - 2.0 * X(3*(j-1)+2) + X(3*(j-2)+2));
            }
            else{
                out(3*j+0) = impl * (X(3*(j-2)) - 4.0 * X(3*(j-1)) + 6.0 * X(3*j) - 4.0 * X(3*(j+1)) + X(3*(j+2)));
                out(3*j+1) = impl * (X(3*(j-2)+1) - 4.0 * X(3*(j-1)+1) + 6.0 * X(3*j+1) - 4.0 * X(3*(j+1)+1) + X(3*(j+2)+1));
                out(3*j+2) = impl * (X(3*(j-2)+2) - 4.0 * X(3*(j-1)+2) + 6.0 * X(3*j+2) - 4.0 * X(3*(j+1)+2) + X(3*(j+2)+2));
            }
    }
    
    return out;    
    
  }


  Vector apply_K(Vector Om, Matrix& U, Matrix& V) {
    //
    //Return result of K w = [sum [t]_x w]
    //
    int Nlk = U.cols();
    Vector out = Vector::Zero(3*Nlk+3);
    
    Eigen::Vector3d U_j, V_j, TxOm;
    
    int offset = 0;
    
    if(!clamp){
        offset = 3;
        out(0) = Om(0);
        out(1) = Om(1);
        out(2) = Om(2);
    }

    
    for (int j = 0; j < Nlk; ++j) {
        // Set components of the frame
        U_j = U.col(j);
        V_j = V.col(j);
        // set tau_cross_Om
        TxOm = Om(offset+1+2*j) * U_j - Om(offset+2*j) * V_j;
        // cumsum of ds*tau_cross_Om
        out(3*(j+1)) = out(3*j) + ds*TxOm(0);
        out(3*(j+1)+1) = out(3*j+1) + ds*TxOm(1);
        out(3*(j+1)+2) = out(3*j+2) + ds*TxOm(2);
    }
    
    return out;
  }

  Vector apply_B(Vector Tension, Matrix& T){
    //
    //Return result of B T = d_s( T t_hat )
    //        
    int Nlk = T.cols();
    Vector out(3*Nlk+3);
    // End points
    int offset;
    if(!clamp){
        offset = 0;
        out(0) =  -Tension(0)*T(0,0);
        out(1) =  -Tension(0)*T(1,0);
        out(2) =  -Tension(0)*T(2,0);
    }
    else {
        offset = 3;
        out(0) =  Tension(0);
        out(1) =  Tension(1);
        out(2) =  Tension(2);
    }
    //
    out(3*Nlk+0) =  Tension(offset+Nlk-1)*T(0,Nlk-1);
    out(3*Nlk+1) =  Tension(offset+Nlk-1)*T(1,Nlk-1);
    out(3*Nlk+2) =  Tension(offset+Nlk-1)*T(2,Nlk-1);
    for (int j = 1; j < Nlk; ++j) {
        // cumsum of ds*tau_cross_Om
        out(3*j+0) =  Tension(offset+j-1)*T(0,j-1)-Tension(offset+j)*T(0,j);
        out(3*j+1) =  Tension(offset+j-1)*T(1,j-1)-Tension(offset+j)*T(1,j);
        out(3*j+2) =  Tension(offset+j-1)*T(2,j-1)-Tension(offset+j)*T(2,j);
    }
    
    return out;
    
  }
  
  Vector Scaled_Impl_Hydro(Matrix& Om, Matrix& Tens, RefMatrix& T, RefMatrix& U, RefMatrix& V, RefVector& X_0){
    //
    // Return result of -M*B*Tens + scale*[I + dt*ds*impl*kb*M*D4]*K*Om
    // Note: the function ds_D4_v actually returns ds*D^4*X
    //            
    const int N_lk = T.cols();
    const int N_fib = T.rows()/3;
    
    const int size_j = 3*N_lk+3;
    const int size = N_fib*size_j;

    Matrix T_j(3,N_lk);
    Matrix U_j(3,N_lk);
    Matrix V_j(3,N_lk);
    Vector X_0_j(3);
    
    Vector BTens(size);
    Vector KOm(size);
    Vector KOm_j;
    Vector D4_KOm(size);
    
    for(int j = 0; j < N_fib; ++j){
        for(int k = 0; k < N_lk; ++k){
            T_j(0,k) = T(3*j+0,k);
            T_j(1,k) = T(3*j+1,k);
            T_j(2,k) = T(3*j+2,k);
            //
            U_j(0,k) = U(3*j+0,k);
            U_j(1,k) = U(3*j+1,k);
            U_j(2,k) = U(3*j+2,k);
            //
            V_j(0,k) = V(3*j+0,k);
            V_j(1,k) = V(3*j+1,k);
            V_j(2,k) = V(3*j+2,k);
            //
            X_0_j(0) = X_0(3*j+0);
            X_0_j(1) = X_0(3*j+1);
            X_0_j(2) = X_0(3*j+2);
        }
        BTens.segment(j*size_j,size_j) = apply_B(Tens.row(j), T_j);
        KOm_j = apply_K(Om.row(j), U_j, V_j);
        KOm.segment(j*size_j,size_j) = KOm_j;
        D4_KOm.segment(j*size_j,size_j) = ds_D4_v(KOm_j);
    }
    
    
    
    Vector mult_by_M = scale*D4_KOm - BTens;
    Vector out = scale*KOm + apply_M(mult_by_M, T, X_0);
    
    return out;
    
  }
  
  
  Vector apply_A_x_Banded_PC(Vector& X, RefMatrix& T, RefMatrix& U, RefMatrix& V, RefVector& X_0, int bands){
    //
    // Apply the Right-PC Operator A*PC(X)
    //
    Matrix Om;
    Matrix Tens;
    
    // Apply PC
    std::tie(Om,Tens) = apply_Banded_PC(X, T, U, V, X_0, bands);
    
    // Apply A
    Om = ((1.0/scale)*Om);
    Vector out = Scaled_Impl_Hydro(Om, Tens, T, U, V, X_0);
    
    //auto A_mult = [&](Matrix Om_in, Matrix Tens_in) -> Vector {return PSE_scaled_Impl_Hydro(Om_in, Tens_in, T, U, V, X_0);};
    //Vector out = A_mult(((1/scale)*Om),Tens);
    return out;
    
  }
  
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

///////////////////////////   TIME INTEGRATION FUNCTIONS

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

  
  
  
  template<class AMatrix, class BVector, class BMatrix>
  void frame_rot(AMatrix& T_all, AMatrix& U_all, AMatrix& V_all, BVector& X_0_all, BMatrix& Om, double delta){
    const int N_fib = T_all.rows()/3;
    const int N_lk = T_all.cols();
    Eigen::Vector3d T_j, U_j, V_j;
    Eigen::Vector3d Omega_j, axis;
    
    double mag_O, theta, s_th, c_th;
    
    // Translate X_0
    if(!clamp){
    for(int k = 0; k < N_fib; ++k){
        for(int d = 0; d < 3; d++){
            X_0_all[3*k+d] += delta*Om(k,d);
        }
    }
    }
    
    //Matrix T(3,N_lk);
    //Matrix U(3,N_lk);
    //Matrix V(3,N_lk);
    
    int offset = 3;
    if(clamp){offset = 0;}
    
    // Rotate the frame
    for(int k = 0; k < N_fib; ++k){
        RefMatrix T = T_all.block(3*k,0,3,N_lk);
        RefMatrix U = U_all.block(3*k,0,3,N_lk);
        RefMatrix V = V_all.block(3*k,0,3,N_lk);
        for(int j = 0; j < N_lk; ++j){
            // Set components of the frame
            T_j = T.col(j);
            U_j = U.col(j);
            V_j = V.col(j);
            // make the rotation axis and magnitude
            Omega_j = Om(k,offset+2*j)*U_j + Om(k,1+offset+2*j)*V_j;
            mag_O = Omega_j.norm();
            theta = delta*mag_O;
            axis = (1.0/mag_O)*Omega_j;
            // Rotate vectors
            s_th = std::sin(theta);
            c_th = std::cos(theta);
            
            T.col(j) = c_th*T_j + s_th*axis.cross(T_j) + (1.0-c_th)*(axis.dot(T_j))*axis;
            T.col(j).normalize();
            U.col(j) = c_th*U_j + s_th*axis.cross(U_j) + (1.0-c_th)*(axis.dot(U_j))*axis;
            U.col(j).normalize();
            V.col(j) = c_th*V_j + s_th*axis.cross(V_j) + (1.0-c_th)*(axis.dot(V_j))*axis;
            V.col(j).normalize();
            //std::cout << T_j.transpose() << '\n';
        }
    }
    

  }
  
  
  Vector M_RFD(RefMatrix& T, RefMatrix& U, RefMatrix& V, RefVector& X_0){
    // RNGesus
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution (0.0,1.0);
    
    const int N_fib = T.rows()/3;
    const int N_lk = T.cols();
    double delta = 1.0e-1*0.5*ds*sqrt(ds);
    
    int size = 3*(N_lk+1);
    Vector W = Vector::Zero(N_fib*size);
    Matrix Om;

    // Make random vector
    for(int k = 0; k < (N_fib*size); ++k){
        W[k] = distribution(generator);
    }
    // Make random velocity
    Om = Kinv_multi(W, U, V);
    
    // Get rotated framesin +- directions
    T_p = T; U_p = U; V_p = V;
    T_m = T; U_m = U; V_m = V;
    X_0_p = X_0; X_0_m = X_0;
    
    //int check = 0;
    //std::cout << "T before: " << T(check+0,0) << " " << T(check+1,0) << " " << T(check+2,0) << "\n";
    frame_rot(T_p, U_p, V_p, X_0_p, Om, (0.5*delta));
    //std::cout << "T after+: " << T(check+0,0) << " " << T(check+1,0) << " " << T(check+2,0) << "\n";
    //std::cout << "Tp after: " << T_p(check+0,0) << " " << T_p(check+1,0) << " " << T_p(check+2,0) << "\n";
    frame_rot(T_m, U_m, V_m, X_0_m, Om, (-0.5*delta));
    //std::cout << "Tm after : " << T_m(check+0,0) << " " << T_m(check+1,0) << " " << T_m(check+2,0) << "\n";
    //std::cout << "T after-: " << T(check+0,0) << " " << T(check+1,0) << " " << T(check+2,0) << "\n";
    
    Vector M_loc_p_W = apply_M(W, T_p, X_0_p);
    Vector M_loc_m_W = apply_M(W, T_m, X_0_m);
    Vector M_RFD = (1.0/delta)*(M_loc_p_W - M_loc_m_W);
    
    return M_RFD;
    
  }  
  
  
  Vector Compute_F(RefMatrix& T_all){
    ////////////////////////////////////////////////
    // Note this actually computes -F
    ///////////////////////////////////////////////
    const int N_fib = T_all.rows()/3;
    const int N_lk = T_all.cols();
    Vector F = Vector::Zero(N_fib*(3+3*N_lk));

    real f_c = k_bend/(ds*ds);
    
    Vector T_j, T_jm1, T_jm2, T_jp1;
    
    
    Matrix T(3,N_lk);
    int offset = 0;
    // Rotate the frame
    for(int k = 0; k < N_fib; ++k){
        offset = k*(3+3*N_lk);
        T = T_all.block(3*k,0,3,N_lk);
        for(int j = 0; j < (N_lk+1); ++j){
            if(j==0){
                if(!clamp){
                    T_j = T.col(j);
                    T_jp1 = T.col(j+1);
                    for(int d = 0; d < 3; d++){
                        F[offset + 3*j + d] = f_c*(T_jp1[d] - T_j[d]);
                    }
                }
            }
            else if(j == N_lk){
                T_j = T.col(j-1);
                T_jm1 = T.col(j-2);
                for(int d = 0; d < 3; d++){
                    F[offset + 3*j + d] = f_c*(T_j[d] - T_jm1[d]);
                }
            }
            else if(j == 1){
                T_j = T.col(j);
                T_jm1 = T.col(j-1);
                T_jp1 = T.col(j+1);
                if(!clamp){
                    for(int d = 0; d < 3; d++){
                        F[offset + 3*j + d] = f_c*(T_jp1[d] - 3.0*T_j[d] + 2.0*T_jm1[d]);
                    }
                }
                else{
                    for(int d = 0; d < 3; d++){
                        F[offset + 3*j + d] = f_c*(T_jp1[d] - 3.0*T_j[d] + 3.0*T_jm1[d] - T_fix[d]);
                    }
                }
            }
            else if(j == (N_lk-1)){
                T_j = T.col(j-1);
                T_jm1 = T.col(j-2);
                T_jp1 = T.col(j);
                for(int d = 0; d < 3; d++){
                    F[offset + 3*j + d] = -1.0*f_c*(T_jm1[d] - 3.0*T_j[d] + 2.0*T_jp1[d]);
                }
            }
            else{
                T_j = T.col(j);
                T_jm1 = T.col(j-1);
                T_jm2 = T.col(j-2);
                T_jp1 = T.col(j+1);
                for(int d = 0; d < 3; d++){
                    F[offset + 3*j + d] = f_c*(T_jp1[d] - 3.0*T_j[d] + 3.0*T_jm1[d] - T_jm2[d]);
                }  
            }
        } //Loop j
    } // Loop k
    
    
    return F;
    
  }
  
  
  auto RHS_and_Midpoint(RefVector& Force, RefMatrix& T, RefMatrix& U, RefMatrix& V, RefVector& X_0){
    
    Vector F = Compute_F(T); // Note this is actually -F
    F -= Force;
    Vector RHS = apply_M(F, T, X_0);
    RHS *= -1.0;
    
    T_h = T;
    U_h = U;
    V_h = V;
    X_0_h = X_0;
    Vector BI;
    
    if(kBT > 1e-10){
        // Make Brownian increment for predictor and corrector steps
        Vector M_half_W1 = apply_Mhalf_W(T, X_0);
        Vector M_half_W2 = apply_Mhalf_W(T, X_0);

        // Make M_RFD
        Vector M_RFD_vec = M_RFD(T, U, V, X_0);
        
        // Set predictor velocity
        double c_1 = 2.0*std::sqrt((kBT/dt));
        Vector BI_half = c_1*M_half_W1;
        Matrix Om_half = Kinv_multi(BI_half, U, V);
        
        // Make RHS for final solve
        BI = std::sqrt((kBT/dt))*(M_half_W1 - M_half_W2);
        RHS += (kBT*M_RFD_vec) + BI;
        

        frame_rot(T_h, U_h, V_h, X_0_h, Om_half, (0.5*dt));
    }
    std::cout << "RHS and Mid OUTPUS THE BI TOO!!!!!!!!!!!!!!!!!!\n";
    return std::make_tuple(RHS, T_h, U_h, V_h, X_0_h, BI);
    
  }
  
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

///////////////////////////   STRESSLET STUFF

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
  
  template<class AVector>
  SparseMd Outer_Product_Mat(AVector& r_vectors, const int N_fib, const int N_blb, bool transpose){
    
    int size = N_fib*(3*N_blb);  
      
    //Matrix Outer_left(9*N_fib,3*N_fib*N_blb);
    //Matrix Outer_right(9*N_fib,3*N_fib*N_blb);
    //Matrix Trace(9*N_fib,3*N_fib*N_blb);
    //Outer_left.setZero();
    //Outer_right.setZero();
    //Trace.setZero();
    
    std::vector<Trip_d> tripletList;
    
    Vector r_k(3);
    Vector COM_j(3);
    
    
    for(int j = 0; j < N_fib; ++j){
        //std::cout << j << "\n";
        COM_j.setZero();
        
        for(int k = 0; k < N_blb; ++k){
            COM_j(0) += (1.0/N_blb)*r_vectors[j*3*N_blb+3*k+0];
            COM_j(1) += (1.0/N_blb)*r_vectors[j*3*N_blb+3*k+1];
            COM_j(2) += (1.0/N_blb)*r_vectors[j*3*N_blb+3*k+2];
        }
        
        
        for(int k = 0; k < N_blb; ++k){
            r_k[0] = r_vectors[j*3*N_blb+3*k+0]-COM_j(0);
            r_k[1] = r_vectors[j*3*N_blb+3*k+1]-COM_j(1);
            r_k[2] = r_vectors[j*3*N_blb+3*k+2]-COM_j(2);
            for(int d = 0; d < 3; ++d){
                /*
                Outer_left(9*j+d+0,j*3*N_blb+3*k+0) = r_k[d];
                Outer_left(9*j+d+3,j*3*N_blb+3*k+1) = r_k[d];
                Outer_left(9*j+d+6,j*3*N_blb+3*k+2) = r_k[d];
                //
                Outer_right(9*j+3*d,j*3*N_blb+3*k) = r_k[d];
                Outer_right(9*j+3*d+1,j*3*N_blb+3*k+1) = r_k[d];
                Outer_right(9*j+3*d+2,j*3*N_blb+3*k+2) = r_k[d]; 
                //
                Trace(9*j+0,j*3*N_blb+3*k+d) = (-1.0/3.0)*r_k[d];
                Trace(9*j+4,j*3*N_blb+3*k+d) = (-1.0/3.0)*r_k[d];
                Trace(9*j+8,j*3*N_blb+3*k+d) = (-1.0/3.0)*r_k[d];
                */
                if(!transpose){
                    tripletList.push_back(Trip_d(9*j+d+0, j*3*N_blb+3*k+0, static_cast<double>(r_k[d])));
                    tripletList.push_back(Trip_d(9*j+d+3, j*3*N_blb+3*k+1, static_cast<double>(r_k[d])));
                    tripletList.push_back(Trip_d(9*j+d+6, j*3*N_blb+3*k+2, static_cast<double>(r_k[d])));
                    // make traceless
                    /*
                    tripletList.push_back(Trip_d(9*j+0,j*3*N_blb+3*k+d, static_cast<double>((-1.0/3.0)*r_k[d])));
                    tripletList.push_back(Trip_d(9*j+4,j*3*N_blb+3*k+d, static_cast<double>((-1.0/3.0)*r_k[d])));
                    tripletList.push_back(Trip_d(9*j+8,j*3*N_blb+3*k+d, static_cast<double>((-1.0/3.0)*r_k[d])));
                    */
                }
                else{
                    tripletList.push_back(Trip_d(j*3*N_blb+3*k+0, 9*j+d+0, static_cast<double>(r_k[d])));
                    tripletList.push_back(Trip_d(j*3*N_blb+3*k+1, 9*j+d+3, static_cast<double>(r_k[d])));
                    tripletList.push_back(Trip_d(j*3*N_blb+3*k+2, 9*j+d+6, static_cast<double>(r_k[d])));
                    // make traceless
                    /*
                    tripletList.push_back(Trip_d(j*3*N_blb+3*k+d, 9*j+0, static_cast<double>((-1.0/3.0)*r_k[d])));
                    tripletList.push_back(Trip_d(j*3*N_blb+3*k+d, 9*j+4, static_cast<double>((-1.0/3.0)*r_k[d])));
                    tripletList.push_back(Trip_d(j*3*N_blb+3*k+d, 9*j+8, static_cast<double>((-1.0/3.0)*r_k[d])));
                    */
                }
                
            }
        }
    }
    
    int Rows = 0;
    int Cols = 0;
    if(!transpose){
        Rows = 9*N_fib;
        Cols = 3*N_fib*N_blb;
    }
    else{
        Rows = 3*N_fib*N_blb;
        Cols = 9*N_fib;
    }
    SparseMd Outer(Rows,Cols);
    Outer.setFromTriplets(tripletList.begin(), tripletList.end());
    
    
    //std::cout << Outer_left << "\n";
    //std::cout << Outer_right << "\n";
    //Matrix Outer = 0.5*(Outer_right + Outer_left) + Trace;
    //Matrix Outer = Outer_left; // + Trace;
    return Outer;
  }

  template<class AMatrix, class BVector>
  Vector Outer_Product_Transpose(BVector& E, AMatrix& T, BVector& X_0){
    std::vector<real> r_vectors = multi_fiber_Pos(T, X_0);
    const int N_lk = T.cols();
    const int N_fib = T.rows()/3;
    const int N_blb = N_lk+1;
    
    SparseMd K_S_T = Outer_Product_Mat(r_vectors, N_fib, N_blb, true);
    //std::cout << K_S_T.rows() << ' ' << K_S_T.cols() << "\n";
    //std::cout << E.rows() << ' ' << E.cols() << "\n";
    Vector E_out = K_S_T * E;
    return E_out;
  }

  template<class AMatrix, class BVector>
  Vector Outer_Product_Tens(AMatrix& Tens, AMatrix& T, BVector& X_0){
    std::vector<real> r_vectors = multi_fiber_Pos(T, X_0);
    const int N_lk = T.cols();
    const int N_fib = T.rows()/3;
    const int N_blb = N_lk+1;
    
    SparseMd K_S = Outer_Product_Mat(r_vectors, N_fib, N_blb, false);
    
    
    const int size_j = 3*N_lk+3;
    const int size = N_fib*size_j;

    Matrix T_j(3,N_lk);
    
    Vector BTens(size);
    
    for(int j = 0; j < N_fib; ++j){
        for(int k = 0; k < N_lk; ++k){
            T_j(0,k) = T(3*j+0,k);
            T_j(1,k) = T(3*j+1,k);
            T_j(2,k) = T(3*j+2,k);
        }
        BTens.segment(j*size_j,size_j) = apply_B(Tens.row(j), T_j);
    }
    Vector S_out = K_S * BTens;
    return S_out;
  }
  
  
  template<class AVector, class AMatrix, class BVector>
  Vector Outer_Product(AVector& Lambda, AMatrix& T, BVector& X_0){
    std::vector<real> r_vectors = multi_fiber_Pos(T, X_0);
    const int N_lk = T.cols();
    const int N_fib = T.rows()/3;
    const int N_blb = N_lk+1;
    
    SparseMd K_S = Outer_Product_Mat(r_vectors, N_fib, N_blb, false);
    
    Vector S_out = K_S * Lambda;
    return S_out;
  }
  
  template<class AMatrix, class BMatrix>
  Vector Apply_K_Multi(AMatrix& Om, BMatrix& U, BMatrix& V){
    //
    // Return result of K*Om
    //            
    const int N_lk = U.cols();
    const int N_fib = U.rows()/3;
    
    const int size_j = 3*N_lk+3;
    const int size = N_fib*size_j;

    Matrix U_j(3,N_lk);
    Matrix V_j(3,N_lk);

    Vector KOm(size);
    Vector KOm_j;
    
    for(int j = 0; j < N_fib; ++j){
        for(int k = 0; k < N_lk; ++k){
            U_j(0,k) = U(3*j+0,k);
            U_j(1,k) = U(3*j+1,k);
            U_j(2,k) = U(3*j+2,k);
            //
            V_j(0,k) = V(3*j+0,k);
            V_j(1,k) = V(3*j+1,k);
            V_j(2,k) = V(3*j+2,k);
        }
        KOm_j = apply_K(Om.row(j), U_j, V_j);
        KOm.segment(j*size_j,size_j) = KOm_j;
    }
    
    return KOm;
  }
  
  template<class AMatrix>
  Matrix K_dense(AMatrix& U, AMatrix& V){
    //
    // Return K as a matrix
    //            
    const int N_lk = U.cols();
    const int N_fib = U.rows()/3;
    
    const int size_j = 3*N_lk+3;
    const int size = N_fib*size_j;
    int size_om = 2*N_lk+3;

    Matrix Om = Matrix::Zero(N_fib,size_om);
    Matrix K = Matrix::Zero(size, size_om);
    
    for(int j = 0; j < N_fib; ++j){
        for(int k = 0; k < size_om; ++k){
            Om(j,k) = 1.0;
            K.col(k) += Apply_K_Multi(Om, U, V);
            Om = Matrix::Zero(N_fib,size_om);
        }
    }
    
    return K;
  }
  
  template<class AVector>
  Matrix M_dense(AVector& pos){
    // mult. M*I
    const int size = pos.size();
    
    Matrix Id = Matrix::Identity(size, size);
    Matrix M(size, size);
    for(int j = 0; j < size; ++j){
      Vector e_j = Id.col(j);
      M.col(j) = apply_M_rvecs(e_j, pos);
    }
    return M;
  }
  
  auto Stresslet_RFD(RefMatrix& T, RefMatrix& U, RefMatrix& V, RefVector& X_0){        
        
    // RNGesus
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution (0.0,1.0);
    
    const int N_fib = T.rows()/3;
    const int N_lk = T.cols();
    double delta = 1.0e-1*0.5*ds*sqrt(ds);
   
//  THIS APPROACH GIVES THE SAME THING AS USING Kinv
// 
//     int size_om = 2*N_lk+3;
//     Matrix Om(N_fib,size_om);
//     Vector Om_v(N_fib*size_om);
// 
//     // Make random vector
//     for(int k = 0; k < N_fib; ++k){
//       for(int j = 0; j < size_om; ++j){
//         Om(k,j) = distribution(generator);
//         Om_v(k*size_om+j) = Om(k,j);
//       }
//     }
    
    ////////////////////
    // deterministic part
    Vector F = Compute_F(T);
    F *= -1.0;
    std::vector<real> pos = multi_fiber_Pos(T, X_0);
    Matrix M = M_dense(pos);
    Matrix K = K_dense(U, V);
    Matrix M_Inv = M.completeOrthogonalDecomposition().pseudoInverse();
    Matrix N_Inv = K.transpose()*M_Inv*K;
    Matrix N = N_Inv.completeOrthogonalDecomposition().pseudoInverse();
    Vector Lambda = M_Inv*K*N*(K.transpose()*F) - F;
    Vector S = Outer_Product(Lambda, T, X_0);
    
    int size = 3*(N_lk+1);
    Vector W = Vector::Zero(N_fib*size);
    Matrix Om;

    // Make random vector
    for(int k = 0; k < (N_fib*size); ++k){
        W[k] = distribution(generator);
    }
    // Make random velocity
    Om = Kinv_multi(W, U, V);
    
    // Get rotated framesin +- directions
    T_p = T; U_p = U; V_p = V;
    T_m = T; U_m = U; V_m = V;
    X_0_p = X_0; X_0_m = X_0;
    
    //int check = 0;
    //std::cout << "T before: " << T(check+0,0) << " " << T(check+1,0) << " " << T(check+2,0) << "\n";
    frame_rot(T_p, U_p, V_p, X_0_p, Om, (0.5*delta));
    std::vector<real> pos_p = multi_fiber_Pos(T_p, X_0_p);
    //std::cout << "T after+: " << T(check+0,0) << " " << T(check+1,0) << " " << T(check+2,0) << "\n";
    //std::cout << "Tp after: " << T_p(check+0,0) << " " << T_p(check+1,0) << " " << T_p(check+2,0) << "\n";
    frame_rot(T_m, U_m, V_m, X_0_m, Om, (-0.5*delta));
    std::vector<real> pos_m = multi_fiber_Pos(T_m, X_0_m);
    //std::cout << "Tm after : " << T_m(check+0,0) << " " << T_m(check+1,0) << " " << T_m(check+2,0) << "\n";
    //std::cout << "T after-: " << T(check+0,0) << " " << T(check+1,0) << " " << T(check+2,0) << "\n";
    
    Matrix M_p = M_dense(pos_p);
    Matrix K_p = K_dense(U_p, V_p);
    Matrix M_p_Inv = M_p.completeOrthogonalDecomposition().pseudoInverse();
    Matrix N_p_Inv = K_p.transpose()*M_p_Inv*K_p;
    Matrix N_p = N_p_Inv.completeOrthogonalDecomposition().pseudoInverse();
    Vector Lambda_p = M_p_Inv*K_p*N_p*(K_p.transpose()*W);
    Vector S_p = Outer_Product(Lambda_p, T_p, X_0_p);
    //std::cout << "N_p\n";
    //std::cout << N_p << "\n";
    
    Matrix M_m = M_dense(pos_m);
    Matrix K_m = K_dense(U_m, V_m);
    Matrix M_m_Inv = M_m.completeOrthogonalDecomposition().pseudoInverse();
    Matrix N_m_Inv = K_m.transpose()*M_m_Inv*K_m;
    Matrix N_m = N_m_Inv.completeOrthogonalDecomposition().pseudoInverse();
    Vector Lambda_m = M_m_Inv*K_m*N_m*(K_m.transpose()*W);
    Vector S_m = Outer_Product(Lambda_m, T_m, X_0_m);
    //std::cout << "N_m\n";
    //std::cout << N_m << "\n";
    
    
    Vector out = (kBT/delta)*(S_p-S_m) + S;
            
    return out;
  }
  
  Vector Stresslet_Strat(std::vector<real>& pos_mid, RefVector& F, const int N_fib, const int N_blb){ 
        SparseMd K_S = Outer_Product_Mat(pos_mid, N_fib, N_blb, false);
        Matrix M_mid = M_dense(pos_mid);
        Matrix M_mid_Inv = M_mid.completeOrthogonalDecomposition().pseudoInverse();
        
        Vector out = K_S*M_mid_Inv*F;
        
        return out;
  }
  
  Vector K_S_RFD(RefMatrix& T, RefMatrix& U, RefMatrix& V, RefVector& X_0){        
 // RNGesus
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution (0.0,1.0);
    
    const int N_fib = T.rows()/3;
    const int N_lk = T.cols();
    double delta = 1.0e-1*0.5*ds*sqrt(ds);
    
    int size = 3*(N_lk+1);
    Vector W = Vector::Zero(N_fib*size);
    Matrix Om;

    // Make random vector
    for(int k = 0; k < (N_fib*size); ++k){
        W[k] = distribution(generator);
    }
    // Make random velocity
    Om = Kinv_multi(W, U, V);
    
    // Get rotated framesin +- directions
    T_p = T; U_p = U; V_p = V;
    T_m = T; U_m = U; V_m = V;
    X_0_p = X_0; X_0_m = X_0;
    
    //int check = 0;
    //std::cout << "T before: " << T(check+0,0) << " " << T(check+1,0) << " " << T(check+2,0) << "\n";
    frame_rot(T_p, U_p, V_p, X_0_p, Om, (0.5*delta));
    frame_rot(T_m, U_m, V_m, X_0_m, Om, (-0.5*delta));
    
    Vector S_p = Outer_Product(W, T_p, X_0_p);
    Vector S_m = Outer_Product(W, T_m, X_0_m);
    
    
    Vector out = (kBT/delta)*(S_p-S_m);
    
    return out;
  }
  
  Vector Stresslet_KsF(RefMatrix& T, RefVector& X_0, RefMatrix& sT, RefVector& sX_0){        
    // Computes K_s*F
    Vector F = Compute_F(T);
    F *= -1.0;
    Vector SF = Outer_Product(F, sT, sX_0);
    return SF;
  }
  
  Vector Stresslet_Correct(RefMatrix& T, RefMatrix& U, RefMatrix& V, RefVector& X_0){        
    // Computes K_s*F
    Vector SF = Stresslet_KsF(T,X_0,T,X_0);
    SF += K_S_RFD(T, U, V, X_0);
    return SF;
  }
  
  auto Kill_Variance(Vector& Mhalf_W, RefMatrix& T, RefMatrix& U, RefMatrix& V, RefVector& X_0){
    std::vector<real> pos = multi_fiber_Pos(T, X_0);
    Matrix M = M_dense(pos);
    Matrix K = K_dense(U, V);
    Matrix KT = K.transpose();
    Matrix M_Inv = M.completeOrthogonalDecomposition().pseudoInverse();
    Matrix N_Inv = KT*M_Inv*K;
    Matrix N = N_Inv.completeOrthogonalDecomposition().pseudoInverse();

    Vector Var_Kill_lambda = (M_Inv*Mhalf_W - M_Inv*K*N*KT*(M_Inv*Mhalf_W));
    Vector S_Var_Kill = Outer_Product(Var_Kill_lambda, T, X_0);
    Vector Var_Kill_lambda_strat = (M_Inv*K*N*KT*(M_Inv*Mhalf_W));
    Vector S_Var_Kill_strat = Outer_Product(Var_Kill_lambda_strat, T, X_0);
    
    return std::make_tuple(S_Var_Kill,S_Var_Kill_strat);
  }
  
  
private:

};



using namespace pybind11::literals;
namespace py = pybind11;
//TODO: Fill python documentation here
PYBIND11_MODULE(c_fibers_obj, m) {
    m.doc() = "Fibers code";
    py::class_<CManyFibers>(m, "CManyFibers").
      def(py::init()).
      def("setParameters", &CManyFibers::setParameters,
	  "Set parameters for the module").
      def("update_T_fix", &CManyFibers::update_T_fix,
      "update ghost tangent").
      def("RHS_and_Midpoint", &CManyFibers::RHS_and_Midpoint,
	  "Generate the RHS for the solve and the midpoint positions").
      def("multi_fiber_Pos", &CManyFibers::multi_fiber_Pos<RefMatrix&, RefVector& >,
	  "Get the blob positions").
      def("frame_rot", &CManyFibers::frame_rot<RefMatrix&, RefVector&, RefMatrix& >,
	  "Rotate the fibers and frame").
      def("apply_A_x_Banded_PC", &CManyFibers::apply_A_x_Banded_PC,
	  "apply A*PC(X)").
      def("apply_Banded_PC", &CManyFibers::apply_Banded_PC,
      "apply the PC").
      def("Outer_Product_Transpose", &CManyFibers::Outer_Product_Transpose<RefMatrix&, RefVector& >,
	  "Calculate the left outer product matrix transpose").
      def("Outer_Product_Tens", &CManyFibers::Outer_Product_Tens<RefMatrix&, RefVector& >,
	  "Calculate the left outer product matrix").
      def("Stresslet_RFD", &CManyFibers::Stresslet_RFD,
	  "Calculate the RFD of the stress").
      def("Stresslet_Strat", &CManyFibers::Stresslet_Strat,
	  "Calculate the Strat integral of the stress").
      def("Stresslet_Correct", &CManyFibers::Stresslet_Correct,
	  "Correct the stress").
      def("Kill_Variance", &CManyFibers::Kill_Variance,
	  "Kill_Variance").
      def("Stresslet_KsF", &CManyFibers::Stresslet_KsF,
	  "Calculate the K_s*F");
}
