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
#include <lapacke.h>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include "sys/time.h"


#include <Eigen/SparseCholesky>


/*
#include"uammd_interface.h"
using PSE = UAMMD_PSE_Glue;
using PSEParameters = PyParameters;
*/

#include"libMobility/solvers/NBody/mobility.h"
#include"libMobility/solvers/NBody_wall/mobility.h"

#include"libMobility/include/MobilityInterface/lanczos.h"

#include"libMobility/solvers/PSE/mobility.h"

#include "DoublyPeriodicStokes/source/gpu/python_wrapper/uammd_interface.h"


using real = libmobility::real;

typedef double real_c;


// Double Typedefs
typedef Eigen::VectorXd Vector;
typedef Eigen::Ref<Vector> RefVector;
typedef Eigen::Ref<const Vector> CstRefVector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::Ref<Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> RefMatrix;

typedef Eigen::Triplet<double> Trip;
typedef Eigen::SparseMatrix<double> SparseM;
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagM;

typedef Eigen::Triplet<double> Trip_d;
typedef Eigen::SparseMatrix<double> SparseMd;

// Rigid types
typedef Eigen::Quaterniond Quat;
typedef Eigen::Matrix3d Matrix3;

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


struct timeval tv;
struct timezone tz;
double timeNow() {
    gettimeofday( &tv, &tz );
    int _mils = tv.tv_usec/1000;
    int _secs = tv.tv_sec;
    return (double)_secs + ((double)_mils/1000.0);
}


struct SpecialParameter{
  real psi = -1.0;
  real Lx = -1, Ly=-1, Lz=-1;
  std::string algorithmForNBody = "default";
};

enum class geometry{rpy, triply_periodic, single_wall, doubly_periodic};
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
    nbw->setParametersNBody_wall(parStar.Lx, parStar.Ly, parStar.Lz);
    solver = nbw;
  }
  else{
  	throw std::runtime_error("This solver is not implemented");
  }
  return solver;
}

auto createSolverDP(PyParameters pyStar, int Np){
    auto dpsolver = std::make_shared<DPStokesGlue>();
    dpsolver->initialize(pyStar, Np);
    return dpsolver;
}


  /*
    mobilityUFRPY computes the 3x3 RPY mobility
    between blobs i and j normalized with 8 pi eta a
  */
  void mobilityUFRPY(real_c rx, real_c ry, real_c rz,
                   real_c &Mxx, real_c &Mxy, real_c &Mxz,
                   real_c &Myy, real_c &Myz, real_c &Mzz,
                   int i, int j,
                   real_c invaGPU){

    real_c fourOverThree = real_c(4.0) / real_c(3.0);

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
      real_c r2 = rx*rx + ry*ry + rz*rz;
      real_c r = std::sqrt(r2);
      //We should not divide by zero but std::numeric_limits<real_c>::min() does not work in the GPU
      //real_c invr = (r > std::numeric_limits<real_c>::min()) ? (real_c(1.0) / r) : (real_c(1.0) / std::numeric_limits<real_c>::min())
      real_c invr = real_c(1.0) / r;
      real_c invr2 = invr * invr;
      real_c c1, c2;
      if(r>=2){
	c1 = real_c(1.0) + real_c(2.0) / (real_c(3.0) * r2);
	c2 = (real_c(1.0) - real_c(2.0) * invr2) * invr2;
	Mxx = (c1 + c2*rx*rx) * invr;
	Mxy = (     c2*rx*ry) * invr;
	Mxz = (     c2*rx*rz) * invr;
	Myy = (c1 + c2*ry*ry) * invr;
	Myz = (     c2*ry*rz) * invr;
	Mzz = (c1 + c2*rz*rz) * invr;
      }
      else{
	c1 = fourOverThree * (real_c(1.0) - real_c(0.28125) * r); // 9/32 = 0.28125
	c2 = fourOverThree * real_c(0.09375) * invr;    // 3/32 = 0.09375
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
  void mobilityUFSingleWallCorrection(real_c rx, real_c ry, real_c rz,
                                      real_c &Mxx, real_c &Mxy, real_c &Mxz,
                                      real_c &Myx, real_c &Myy, real_c &Myz,
                                      real_c &Mzx, real_c &Mzy, real_c &Mzz,
                                      int i, int j,
                                      real_c hj){

    if(i == j){
      real_c invZi = real_c(1.0) / hj;
      real_c invZi3 = invZi * invZi * invZi;
      real_c invZi5 = invZi3 * invZi * invZi;
      Mxx += -(9*invZi - 2*invZi3 + invZi5 ) / real_c(12.0);
      Myy += -(9*invZi - 2*invZi3 + invZi5 ) / real_c(12.0);
      Mzz += -(9*invZi - 4*invZi3 + invZi5 ) / real_c(6.0);
    }
    else{
      real_c h_hat = hj / rz;
      real_c invR = 1.0/std::sqrt(rx*rx + ry*ry + rz*rz); // = 1 / r; //TODO: Make this a fast inv sqrt
      real_c ex = rx * invR;
      real_c ey = ry * invR;
      real_c ez = rz * invR;
      real_c invR3 = invR * invR * invR;
      real_c invR5 = invR3 * invR * invR;

      real_c fact1 = -(3*(1+2*h_hat*(1-h_hat)*ez*ez) * invR + 2*(1-3*ez*ez) * invR3 - 2*(1-5*ez*ez) * invR5)  / real_c(3.0);
      real_c fact2 = -(3*(1-6*h_hat*(1-h_hat)*ez*ez) * invR - 6*(1-5*ez*ez) * invR3 + 10*(1-7*ez*ez) * invR5) / real_c(3.0);
      real_c fact3 =  ez * (3*h_hat*(1-6*(1-h_hat)*ez*ez) * invR - 6*(1-5*ez*ez) * invR3 + 10*(2-7*ez*ez) * invR5) * real_c(2.0) / real_c(3.0);
      real_c fact4 =  ez * (3*h_hat*invR - 10*invR5) * real_c(2.0) / real_c(3.0);
      real_c fact5 = -(3*h_hat*h_hat*ez*ez*invR + 3*ez*ez*invR3 + (2-15*ez*ez)*invR5) * real_c(4.0) / real_c(3.0);

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




class CManyBodies{
  real_c a, dt, kBT, eta;
  Vector Lp;
  //PSEParameters par, par_Mhalf;
  std::shared_ptr<libmobility::Mobility> solver;
  std::shared_ptr<DPStokesGlue> DPsolver;
  std::shared_ptr<DPStokesGlue> DPsolver_PC;
  std::shared_ptr<LanczosStochasticDisplacements> lanczos; 
  PyParameters pyStar;
  SpecialParameter parStar;
  geometry geom; //geometry::single_wall; //geometry::rpy; //geometry::triply_periodic; //
  int DomainInt;
  
  // Solver parameters
  bool PC_wall = false; // use wall corrections in PC or not
  bool block_diag_PC = false;
  double M_scale;
  bool split_rand = false;
  bool PC_mat_Set = false;
  SparseM invM;
  SparseM Ninv;
  std::vector<Eigen::LLT<Matrix>> N_lu;
  
  // rigid coordinates
  bool cfg_set = false;
  int N_bod;
  // Solver config
  std::vector<Quat> Qs;
  std::vector<Vector> Xs;
  // Base config
  std::vector<Quat> Q_n;
  std::vector<Vector> X_n;
  
  // Body configurations
  Matrix Ref_Cfg;
  int N_blb;
  bool parametersSet = false;
  SparseM K, KT, Kinv;
  
  //tbb::global_control tbbControl;
public:
  // CBodies():
  //   tbbControl(tbb::global_control::max_allowed_parallelism, 3){     
  // }

  void setParametersPSE(real psi){
    parStar.psi = psi;
  }
  
  void setParametersDP(int nx, int ny, int nz, real Lx, real Ly,
                       real zmin, real zmax, real w, real w_d, 
                       real beta, real beta_d, std::string mode){
    pyStar.nx = nx;
    pyStar.ny = ny;
    pyStar.nz = nz;
    pyStar.Lx = Lx;
    pyStar.Ly = Ly;
    pyStar.zmin = zmin;
    pyStar.zmax = zmax;
    pyStar.w = w;
    pyStar.w_d = w_d;
    pyStar.beta = beta;
    pyStar.beta_d = beta_d;
    pyStar.mode = mode;
  }
  
  
  
  void removeMean(Matrix& Cfg){
      Vector mean = Cfg.colwise().mean();
      std::cout << "Old mean: " << mean.transpose() << "\n";
      for (int i = 0; i < Cfg.rows(); ++i){
        Cfg.row(i) = Cfg.row(i) - mean.transpose();
      }

  }
    
  void setParameters(int numParts, real_c a, real_c dt, real_c kBT, real_c eta, Vector Lp, Matrix& Cfg, int DomainInt){
    // TODO: Put the list of parameters into a structure
    this->a = a;
    this->dt = dt;
    this->kBT = kBT;
    this->eta = eta;
    this->Lp = Lp;
    this->DomainInt = DomainInt;
    removeMean(Cfg);
    
    std::cout << "New mean of Ref Config canged to: " << (Cfg.colwise().mean()).transpose() << "\n";
    
    this->Ref_Cfg = Cfg;
    this->N_blb = Ref_Cfg.rows();
    this->parametersSet = true;
    
    this->M_scale = 1.0; //(6.0*M_PI*eta*a); //(6.0*M_PI*eta*a);
        
    if(DomainInt == 0){
        geom = geometry::rpy;
    }
    else if(DomainInt == 1){
        geom = geometry::single_wall;
    }
    else if(DomainInt == 2){
        geom = geometry::doubly_periodic;
    }
    else if(DomainInt == 3){
        geom = geometry::triply_periodic;
    }
    else{
        std::cout << "using built in rpy implementation\n";
    }
    
    if(DomainInt < 2){
      libmobility::Parameters par;
      par.hydrodynamicRadius = {a};
      par.viscosity = eta;
      par.temperature = 0.0;
      par.numberParticles = numParts;
      parStar.Lx = Lp[0]; parStar.Ly = Lp[1]; parStar.Lz = Lp[2];
      solver = createSolver(geom, parStar);
      solver->initialize(par);
    }
    if(geom == geometry::doubly_periodic){
      pyStar.viscosity = eta;
      pyStar.hydrodynamicRadius = a;
    }
        
    
    

  }
  
  void setBlkPC(bool PCtype){
      block_diag_PC = PCtype;
  }
  
  void setWallPC(bool Wall){
      PC_wall = Wall;
  }
  
  
  void setConfig(RefVector& X_0, RefVector& Q){
      N_bod = X_0.size()/3;
      Qs.reserve(4*N_bod);
      Xs.reserve(3*N_bod);
      Q_n.reserve(4*N_bod);
      X_n.reserve(3*N_bod);
      
      for(int j = 0; j < N_bod; ++j){
        Quat Q_j;
        Vector X_0_j(3);
        // set quaternion
        Q_j.x() = Q(4*j+1);
        Q_j.y() = Q(4*j+2);
        Q_j.z() = Q(4*j+3);
        Q_j.w() = Q(4*j+0);
        Q_j.normalize();
        Qs.push_back(Q_j);
        Q_n.push_back(Q_j);
        // set disp
        X_0_j(0) = X_0(3*j+0);
        X_0_j(1) = X_0(3*j+1);
        X_0_j(2) = X_0(3*j+2);
        Xs.push_back(X_0_j);
        X_n.push_back(X_0_j);
      }
      cfg_set = true;
  }
  
  
  std::tuple<Vector,Vector> getConfig(){
      Vector Qout(4*N_bod);
      Vector Xout(3*N_bod);
      Quat Q_j;
      Vector X_0_j(3);
      for(int j = 0; j < N_bod; ++j){
        Q_j = Qs[j];
        // set quaternion
        Qout(4*j+1) = Q_j.x();
        Qout(4*j+2) = Q_j.y();
        Qout(4*j+3) = Q_j.z();
        Qout(4*j+0) = Q_j.w();
        // set disp
        X_0_j = Xs[j];
        Xout(3*j+0) = X_0_j(0);
        Xout(3*j+1) = X_0_j(1);
        Xout(3*j+2) = X_0_j(2);
      }
      
      return std::make_tuple(Qout,Xout);
  }
  
  
  Matrix get_r_vecs(Vector& X_0, Quat& Q){
    // ...
    Matrix3 rotation_matrix = Q.toRotationMatrix();
    //std::cout << rotation_matrix << '\n';
    Matrix r_vectors = Ref_Cfg * rotation_matrix.transpose();
    Vector r_j;
    for(int j = 0; j < N_blb; ++j){
        r_vectors.row(j) += X_0;
    }
    return r_vectors;

  }
  
  
  std::vector<real_c> single_body_pos(Vector& X_0, Quat& Q){
    // ...
    Matrix r_vectors = get_r_vecs(X_0, Q);
    Vector r_j;
    std::vector<real_c> pos;
    pos.reserve(3*N_blb);
    for(int j = 0; j < N_blb; ++j){
        r_j = r_vectors.row(j);
        pos.push_back(r_j(0));
        pos.push_back(r_j(1));
        pos.push_back(r_j(2));
    }
    return pos;

  }
  
  std::vector<real_c> r_vecs_from_cfg(std::vector<Vector>& Xin, std::vector<Quat>& Qin){
    int size = N_bod*(Ref_Cfg.size());
    std::vector<real_c> pos;
    pos.reserve(size);
    std::vector<real_c> pos_j;
    for(int j = 0; j < N_bod; ++j){
        pos_j = single_body_pos(Xin[j], Qin[j]);
        pos.insert( pos.end(), pos_j.begin(), pos_j.end() );
        pos_j.clear();
    }
    return pos;

  }
  
  std::vector<real_c> multi_body_pos(){
    if(!cfg_set){
        std::cout << "ERROR CONFIG NOT INITIALIZED YET!!\n";
    }
    return r_vecs_from_cfg(Xs, Qs);

  }

  
  Matrix block_KTKinv(double sumr2_cfg, Matrix3& MOI_cfg, Quat& Q_j){
    Matrix KTKinv(6,6);
    KTKinv.setZero();
    int N = N_blb;
    Matrix3 Ainv = (1.0/(1.0 * N)) * Matrix::Identity(3,3);
    Matrix3 B;
    B.setZero();
    Matrix3 C = B.transpose();
    
    Matrix3 Rot = Q_j.toRotationMatrix();
    Matrix3 D = (sumr2_cfg) * Matrix::Identity(3,3) - Rot*MOI_cfg*Rot.transpose();   
    
    double D_det = D.determinant();
    if(D_det < 1.0e-13){
       std::cout << "ERROR K^{T}*K IS SIGULAR (is your rigid body a dimer?)\n";
       exit (EXIT_FAILURE);
    }
    
    
    Matrix3 S = D.inverse(); // = (D - C*Ainv*B)^-1;
    
    KTKinv.block<3,3>(0,0) = Ainv; //Ainv + Ainv*B*S*C*Ainv;
    //KTKinv.block<3,3>(0,3) = -1.0*Ainv*B*S;
    //KTKinv.block<3,3>(3,0) = -1.0*S*C*Ainv;
    KTKinv.block<3,3>(3,3) = S;
    
    //std::cout << "Sinv: " << Sinv << "\nC-C*Ainv*B: " << (D - C*Ainv*B) << "\n";

    
    /*
    Matrix KTK(6,6);
    KTK.block<3,3>(0,0) = (1.0 * N) * Matrix::Identity(3,3);
    KTK.block<3,3>(0,3) = B;
    KTK.block<3,3>(3,0) = C;
    KTK.block<3,3>(3,3) = D;
    
    std::cout << "MATRIX NORM ERROR:\n";
    std::cout << (KTK*KTKinv -  Matrix::Identity(6,6)).squaredNorm() << "\n";
    */
    
    
    return KTKinv;
  }
  
  std::tuple<SparseM,SparseM> Make_K_Kinv(std::vector<Vector> &Xin, std::vector<Quat> &Qin){
    Matrix r_vectors;
    Vector r_k(3);
    //
    SparseM K_mat(3*N_bod*N_blb,6*N_bod);
    SparseM KTKi_mat(6*N_bod,6*N_bod);
    
    std::vector<Trip> tripletList;
    std::vector<Trip> tripletList_KTKi;
    tripletList.reserve(9*N_blb*N_bod);
    tripletList_KTKi.reserve(6*6*N_bod);
    
    int offset = 0;
    
    // sum of r_i^{T}*r_i and sum of r_i*r_i^{T}
    // where r_i is in Ref config (for (KT*K)^-1)
    double sumr2_cfg = Ref_Cfg.squaredNorm();
    Matrix3 MOI_cfg; 
    MOI_cfg *= 0.0;
    for(int k = 0; k < N_blb; ++k){
        r_k = Ref_Cfg.row(k);
        MOI_cfg += r_k * r_k.transpose();
    }
    
    Matrix KTKi_block(6,6);
    
    for(int j = 0; j < N_bod; ++j){
        
        r_vectors = get_r_vecs(Xin[j], Qin[j]);
        
        
        // Set blocks of (K^T*K)^-1
        KTKi_block = block_KTKinv(sumr2_cfg,MOI_cfg,Qin[j]);
        
        for(int rw = 0; rw < 6; ++rw){
            for(int cl = 0; cl < 6; ++cl){
                tripletList_KTKi.push_back(Trip(6*j+rw, 6*j+cl, KTKi_block(rw,cl)));
            }
        }
        
        
        // set blocks of K
        for(int k = 0; k < N_blb; ++k){
            tripletList.push_back(Trip(offset+3*k+0, 6*j+0, 1.0));
            tripletList.push_back(Trip(offset+3*k+1, 6*j+1, 1.0));
            tripletList.push_back(Trip(offset+3*k+2, 6*j+2, 1.0));
            //
            r_k = r_vectors.row(k) - Xin[j].transpose();
            //
            tripletList.push_back(Trip(offset+3*k+0, 6*j+4, r_k(2)));
            tripletList.push_back(Trip(offset+3*k+0, 6*j+5, -r_k(1)));
            tripletList.push_back(Trip(offset+3*k+1, 6*j+5, r_k(0)));
            
            tripletList.push_back(Trip(offset+3*k+1, 6*j+3, -r_k(2)));
            tripletList.push_back(Trip(offset+3*k+2, 6*j+3, r_k(1)));
            tripletList.push_back(Trip(offset+3*k+2, 6*j+4, -r_k(0)));
        }
        offset += 3*N_blb;
        
    }
    K_mat.setFromTriplets(tripletList.begin(), tripletList.end());

    
    KTKi_mat.setFromTriplets(tripletList_KTKi.begin(), tripletList_KTKi.end());
    
    SparseM KI = (KTKi_mat*K_mat.transpose()).pruned();
    
    
    return std::make_tuple(K_mat,KI);
  }
  
  
  void set_K_mats(){
    //
    if(!cfg_set){
        std::cout << "ERROR CONFIG NOT INITIALIZED YET!!\n";
    }  
      
    std::tie(K,Kinv) = Make_K_Kinv(Xs, Qs);
    KT = K.transpose();
    
    
    
//     Vector U(6*N_bod);
//     U.setOnes();
//     Vector KU = K*U;
//     std::cout << "error Kinv*K*U: " << (Kinv*KU-U).norm() << "\n";

    
//      SparseM KTK = (KT * K);
//      std::cout << "MATRIX NORM ERROR:\n";
//      std::cout << (KTK*KTKi_mat) << "\n";
    
  }

  Vector K_x_U(RefVector& U){
     return K*U;
  }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

///////////////////////////   PRECONDITIONER/Solver FUNCTIONS

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  
  
  template<class AVector>
  Matrix rotne_prager_tensor(AVector& r_vectors){

    // Compute scalar functions f(r) and g(r)
    real_c norm_fact_f = real_c(1.0) / (8.0 * M_PI * eta * a);

    // Build mobility matrix of size 3N \times 3N
    int N = r_vectors.size();
    int Nparts = N/3;
    
    real_c invaGPU = real_c(1.0) / a;
    
    real_c rx, ry, rz;

    real_c Mxx, Mxy, Mxz;
    real_c Myx, Myy, Myz;
    real_c Mzx, Mzy, Mzz;
    
    Matrix Mob(N,N);
    Matrix3 Block;
    
    for (int i = 0; i < Nparts; ++i) {
        for (int j = i; j < Nparts; ++j) {
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
            
            Block << Mxx, Mxy, Mxz,
                     Myx, Myy, Myz,
                     Mzx, Mzy, Mzz;
                     
            Mob.block<3,3>(3*i,3*j) = Block;
            if(j != i){Mob.block<3,3>(3*j,3*i) = Block.transpose();}
        }
    }
    
    Mob *= norm_fact_f;
    
    return Mob;
  }
  
  SparseM Block_diag_invM(){
      
    int Blk_sz = 3*N_blb;  

    SparseM Blk_Mob(Blk_sz*N_bod,Blk_sz*N_bod);
    Matrix Mob(Blk_sz,Blk_sz);
    Matrix Minv(Blk_sz,Blk_sz);
    
    std::vector<Trip> tripletList;
    tripletList.reserve(Blk_sz*Blk_sz*N_bod);
    
    std::vector<real_c> r_vectors; 
    
    //double t, elapsed;
    
    for (int i = 0; i < N_bod; ++i) {
        
        r_vectors = single_body_pos(Xs[i], Qs[i]);
        //t = timeNow();
        Mob = rotne_prager_tensor(r_vectors); //Dense_M(r_vectors); //
        //elapsed = timeNow() - t;
        //printf( "Mob time = %g\n", elapsed );
        //t = timeNow();
        Minv = Mob.inverse();
        //elapsed = timeNow() - t;
        //printf( "Inv time = %g\n", elapsed );
        
        for(int rw = 0; rw < Blk_sz; ++rw){
            for(int cl = 0; cl < Blk_sz; ++cl){
                tripletList.push_back(Trip(Blk_sz*i+rw, Blk_sz*i+cl, Minv(rw,cl)));
            }
        }
        

    }
    Blk_Mob.setFromTriplets(tripletList.begin(), tripletList.end());
    
    //std::cout << mat << "\n";
    
    return Blk_Mob;
  }
  
  
  template<class AVector>
  SparseM diag_invM(AVector& r_vectors){

    // INVERSE
    real_c norm_fact_f = real_c(8.0 * M_PI * eta * a);

    // Build mobility matrix of size 3N \times 3N
    int N = r_vectors.size();
    int Nparts = N/3;
    
    real_c invaGPU = real_c(1.0) / a;
    
    real_c rx=0;
    real_c ry=0;
    real_c rz=0;

    real_c Mxx, Mxy, Mxz;
    real_c Myx, Myy, Myz;
    real_c Mzx, Mzy, Mzz;
    
    std::vector<Trip> tripletList;
    tripletList.reserve(Nparts*3*3);
    
    Matrix3 Block, Minv;
    
    for (int i = 0; i < Nparts; ++i) {
        int j = i;

        mobilityUFRPY(rx,ry,rz, Mxx,Mxy,Mxz,Myy,Myz,Mzz, i,j, invaGPU);
        Myx = Mxy;
        Mzx = Mxz;
        Mzy = Myz;
        if(PC_wall){
            mobilityUFSingleWallCorrection(rx/a, ry/a, (rz+2*r_vectors[3*j+2])/a, Mxx,Mxy,Mxz,Myx,Myy,Myz,Mzx,Mzy,Mzz, i,j, r_vectors[3*j+2]/a);
        }
        
        Block << Mxx, Mxy, Mxz,
                 Myx, Myy, Myz,
                 Mzx, Mzy, Mzz;
                 
        Minv = Block.inverse();
        
        tripletList.push_back(Trip(i * 3, j * 3, Minv(0,0)));
        tripletList.push_back(Trip(i * 3, j * 3 + 1, Minv(0,1)));
        tripletList.push_back(Trip(i * 3, j * 3 + 2, Minv(0,2)));

        tripletList.push_back(Trip(i * 3 + 1, j * 3, Minv(1,0)));
        tripletList.push_back(Trip(i * 3 + 1, j * 3 + 1, Minv(1,1)));
        tripletList.push_back(Trip(i * 3 + 1, j * 3 + 2, Minv(1,2)));

        tripletList.push_back(Trip(i * 3 + 2, j * 3, Minv(2,0)));
        tripletList.push_back(Trip(i * 3 + 2, j * 3 + 1, Minv(2,1)));
        tripletList.push_back(Trip(i * 3 + 2, j * 3 + 2, Minv(2,2)));
    }
    SparseM mat(N,N);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    
    mat *= norm_fact_f;
    
    //std::cout << mat << "\n";
    
    return mat;
  }
  
  
  SparseM PC_invM(){
      std::cout << "Making PC mats\n";
      if(!block_diag_PC){
        std::vector<real_c> r_vectors = multi_body_pos();
        return diag_invM(r_vectors);
      }
      else{
        std::cout << "using Block diag PC\n";
        return Block_diag_invM();   
      }  
  }
  
  std::vector<Eigen::LLT<Matrix>> get_blk_diag_lu(SparseM& Ninv){
      Matrix Block(6,6);
      
      std::vector<Eigen::LLT<Matrix>> N_lu;
      N_lu.reserve(N_bod*6*6);
      
      for(int i = 0; i < N_bod; ++i){
          Block = Ninv.block(6*i,6*i,6,6);
          Eigen::LLT<Matrix> lu(Block);
          N_lu.push_back(lu);
      }
      
      return N_lu;
  }
  
  
  template<class AVector>
  void test_PC(AVector& Lambda, AVector& U){
      
      SparseM invM = PC_invM();
      
      Eigen::SimplicialLLT<SparseM> chol(invM);
      Vector Slip = chol.solve(Lambda) - K*U;
      Vector F = -KT*Lambda;
      
      Vector IN(3*N_bod*N_blb + 6*N_bod);
      IN << Slip, F;
      Vector OUT = apply_PC(IN);
      Vector Lpc = OUT.head(3*N_bod*N_blb);
      Vector Upc = OUT.tail(6*N_bod);
      
      
      Lpc *= (1.0/M_scale);
      
      std::cout << "error Lambda: " << (Lpc-Lambda).norm() << "\n";
      std::cout << "error U: " << (Upc-U).norm() << "\n";
      
  }
  
  
  template<class AVector>
  Vector apply_PC(AVector& IN){
      
      if( not PC_mat_Set){
        invM = PC_invM();
        Ninv = (KT*invM*K).pruned();
        N_lu = get_blk_diag_lu(Ninv);
        PC_mat_Set = true;
      }
      
      Vector Slip = IN.head(3*N_bod*N_blb);
      Vector F = IN.tail(6*N_bod);
      
      Vector RHS = -F - KT*(invM*Slip);
      
      //double t, elapsed;
      
      //t = timeNow();
      
      //elapsed = timeNow() - t;
      //printf( "Block decompose time = %g\n", elapsed );
                  
      //t = timeNow();
      Vector b(6);
      Vector U(6*N_bod);
      for(int i = 0; i < N_bod; ++i){
          b = RHS.segment<6>(6*i);
          U.segment<6>(6*i) = N_lu[i].solve(b);
      }
      //elapsed = timeNow() - t;
      //printf( "Block solve time = %g\n", elapsed );
      
      // SPARSE SOLVE IS SLOWER TO FACTOR. SAME TO SOLVE
      /* 
      static Eigen::SimplicialLLT<SparseM> chol(Ninv);
      elapsed = timeNow() - t;
      printf( "Sparse decompose time = %g\n", elapsed );
      t = timeNow();
      Vector temp = chol.solve(RHS);
      elapsed = timeNow() - t;
      printf( "Sparse solve time = %g\n", elapsed );
      */
      
      //std::cout << "U is: " << U << "\n";
      //std::cout << "solve U is: " << chol.solve(RHS) << "\n";
      
      Vector Lambda = M_scale*(invM*(Slip + K*U));
      
      Vector Out(3*N_bod*N_blb + 6*N_bod);
      Out << Lambda, U;
      
      return Out;
      
  }
  
  DiagM make_damp_mat(std::vector<real_c>& r_vectors){        
        int N = r_vectors.size();
        int Nparts = N/3;
        Vector B(N);
        double d_ii;
        
        for(int i = 0; i < Nparts; ++i){
           if(r_vectors[3*i+2] >= a){
             d_ii = 1.0;    
           }
           else{
             d_ii = r_vectors[3*i+2]/a;
           }
           for(int k = 0; k <3; ++k){
             B(3*i+k) = d_ii;  
           }
        }
        return B.asDiagonal();
  }
  
  
  template<class AVector>
  Vector apply_M(AVector& F, std::vector<real_c>& r_vectors){
        int sz = r_vectors.size();
        Vector U(sz);
        U.setZero();
        if(geom == geometry::doubly_periodic){
            if(not DPsolver){
                int NP = sz/3;
                DPsolver = createSolverDP(pyStar,NP);
            }
            DPsolver->setPositions(r_vectors.data());
            DPsolver->Mdot(F.data(),nullptr,U.data(),nullptr);
        }
        else{
            if(DomainInt < 4){
                //Matrix M = rotne_prager_tensor(r_vectors);
                solver->setPositions(r_vectors.data());
                solver->Mdot(F.data(),nullptr,U.data());
            }
            else{
                Matrix M = rotne_prager_tensor(r_vectors);
                DiagM B = make_damp_mat(r_vectors);
                U = B * (M * (B * F));
            }
        }
        return U;
  }
  
  template<class AVector>
  Vector apply_M_PC(AVector& F, std::vector<real_c>& r_vectors){
        int sz = r_vectors.size();
        Vector U(sz);
        U.setZero();
        if(geom == geometry::doubly_periodic){
            if(not DPsolver_PC){
                int NP = sz/3;
                DPsolver_PC = createSolverDP(pyStar,NP);
            }
            DPsolver_PC->setPositions(r_vectors.data());
            DPsolver_PC->Mdot(F.data(),nullptr,U.data(),nullptr);
        }
        else{
            if(DomainInt < 4){
                //Matrix M = rotne_prager_tensor(r_vectors);
                solver->setPositions(r_vectors.data());
                solver->Mdot(F.data(),nullptr,U.data());
            }
            else{
                Matrix M = rotne_prager_tensor(r_vectors);
                DiagM B = make_damp_mat(r_vectors);
                U = B * (M * (B * F));
            }
        }
        return U;
  }
  
  Matrix Dense_M(std::vector<real_c>& r_vectors){
    int Blk_sz = r_vectors.size();

    Matrix Mob(Blk_sz,Blk_sz);
    Matrix Id = Matrix::Identity(Blk_sz,Blk_sz);
    Vector e_i(Blk_sz);
    
    for (int i = 0; i < N_blb; ++i) {
        e_i = Id.col(i);
        Mob.col(i) = apply_M_PC(e_i,r_vectors);
    }
    return Mob;
  }
  
  
  Vector M_half_W(){        
        std::vector<real_c> r_vectors = multi_body_pos();
        int sz = 3*N_bod*N_blb;
        // Make random vector
        Vector W = rand_vector(sz);
        Vector Out(sz);
        if(geom == geometry::doubly_periodic){
            if(not DPsolver){
                DPsolver = createSolverDP(pyStar,(N_bod*N_blb));
            }
            if(not lanczos){
                real tol = 1.0e-4;
                lanczos = std::make_shared<LanczosStochasticDisplacements>((N_bod*N_blb), 1.0, tol);
            } 
            DPsolver->setPositions(r_vectors.data());
            lanczos->stochasticDisplacements([this](const real*f, real*mv){DPsolver->Mdot(f, nullptr, mv, nullptr);}, Out.data(), 1.0); 
        }
        else{
            if(DomainInt < 4){
                solver->setPositions(r_vectors.data());
                solver->stochasticDisplacements(Out.data(), 1.0);
            }
            else{
                Matrix Mob = rotne_prager_tensor(r_vectors);
                DiagM B = make_damp_mat(r_vectors);
                Matrix M = B*Mob*B;
                Eigen::LLT<Matrix> chol(M);
                Matrix L = chol.matrixL();
                Out = (L*W);
            }
            
        }
        
        //std::cout << "r_vecs: " << r_vectors[12] << "\n";
        //std::cout << "L: " << L.block<3,3>(24,0) << "\n";
                
        return Out;
  }
  
  
  
  template<class AVector>
  Vector apply_Saddle(AVector& IN){
      std::vector<real_c> r_vectors = multi_body_pos();
      
      Vector Lambda = IN.head(3*N_bod*N_blb);
      Vector U = IN.tail(6*N_bod);
      
      Vector Slip = M_scale*apply_M(Lambda,r_vectors) - K*U;
      Vector F = -KT*Lambda;
      
      Vector Out(3*N_bod*N_blb + 6*N_bod);
      Out << Slip, F;
            
      return Out;
  }
  
  
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

///////////////////////////   Dynamics/time integration

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
    Quat Q_from_Om(Vector& Om){
        // ...
        Quat Q_rot = Quat::Identity();
        double Om_norm = Om.norm();
        Q_rot.w() = cos(Om_norm/2.0);
        if(Om_norm > 1.0e-10){
            Q_rot.vec() = (sin(Om_norm/2.0)/Om_norm)*Om;
        }
        Q_rot.normalize();
        return Q_rot;
    }
    
    
    std::tuple<std::vector<Quat>,std::vector<Vector>> update_X_Q(Vector& U){
        std::vector<Quat> Qin( Q_n );
        std::vector<Vector> Xin( X_n );

        Quat Q_rot;
        Vector U_j(3);
        Vector Om_j(3);
        
        for(int j = 0; j < N_bod; ++j){
            U_j = U.segment<3>(6*j);
            Om_j = U.segment<3>(6*j+3);
            // set quaternion
            Q_rot = Q_from_Om(Om_j);
            // Update 
            Qin[j] = Q_rot*Qin[j];
            Qin[j].normalize();
            Xin[j] = Xin[j] + U_j;
        }
        
        return std::make_tuple(Qin,Xin);
    }
    
    
    std::tuple<Matrix, Matrix> update_X_Q_out(Vector& U){
        std::vector<Quat> Qin;
        std::vector<Vector> Xin;
        std::tie(Qin,Xin) = update_X_Q(U);
        
        Matrix Qout(N_bod,4);
        Matrix Xout(N_bod,3);
        for(int j = 0; j < N_bod; ++j){
            Qout(j,0) = Qin[j].w();
            Qout(j,1) = Qin[j].x();
            Qout(j,2) = Qin[j].y();
            Qout(j,3) = Qin[j].z();
            //
            Xout.row(j) = Xin[j];
        }
        return std::make_tuple(Qout,Xout);
    }
    
    
    Vector rand_vector(int N){
        // RNGesus
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        std::normal_distribution<double> distribution (0.0,1.0);
        
        Vector W = Vector::Zero(N);
        // Make random vector
        for(int k = 0; k < N; ++k){
            W[k] = distribution(generator);
        }
        
        return W;
        
        
    }
    
    Vector KTinv_RFD(){        
        
        
        double delta = 1.0e-4;
        
        // Make random vector
        Vector W = rand_vector(6*N_bod);
        
        
        std::vector<Quat> Qp;
        std::vector<Vector> Xp;
        std::vector<Quat> Qm;
        std::vector<Vector> Xm;
        
        
        Vector Win = ((delta/2.0)*W);
        std::tie(Qp,Xp) = update_X_Q(Win);
        Win *= -1.0;
        std::tie(Qm,Xm) = update_X_Q(Win);
        
        SparseM Kp, Kinvp, Km, Kinvm;
        std::tie(Kp,Kinvp) = Make_K_Kinv(Xp, Qp);
        std::tie(Km,Kinvm) = Make_K_Kinv(Xm, Qm);
        
        Vector out = (1.0/delta)*Kinvp.transpose()*W - (1.0/delta)*Kinvm.transpose()*W;
        
        std::cout << out << "n";
        
        return (KT*out);
    }
    
    Vector M_RFD(){        
        
        double delta = 1.0e-4;
        
        int sz = 3*N_bod*N_blb;
       // Make random vector
        Vector W = rand_vector(sz);
        
        Vector UOM = Kinv*W;
        
        std::vector<Quat> Qp;
        std::vector<Vector> Xp;
        std::vector<Quat> Qm;
        std::vector<Vector> Xm;
        
        
        Vector Win = ((delta/2.0)*UOM);
        std::tie(Qp,Xp) = update_X_Q(Win);
        std::vector<real_c> r_vec_p =  r_vecs_from_cfg(Xp,Qp);
        Win *= -1.0;
        std::tie(Qm,Xm) = update_X_Q(Win);
        std::vector<real_c> r_vec_m =  r_vecs_from_cfg(Xm,Qm);
        
        Vector Mp = apply_M(W, r_vec_p);
        Vector Mm = apply_M(W, r_vec_m);
        
        Vector out = (1.0/delta)*(Mp-Mm);
                
        return out;
    }
    
    void evolve_X_Q(Vector& U){
        std::vector<Quat> Qnext;
        std::vector<Vector> Xnext;
        
        U *= dt;
        
        std::tie(Qnext,Xnext) = update_X_Q(U);
        
        Q_n = Qnext;
        X_n = Xnext;
        Qs = Qnext;
        Xs = Xnext;
        
        set_K_mats();
        PC_mat_Set = false;
    }
    
    
    Vector Test_Mhalf(int N){
        Vector error(N);
        std::vector<real_c> r_vectors = multi_body_pos();
        Matrix M = rotne_prager_tensor(r_vectors);
        Matrix M_rand(3*N_bod*N_blb,3*N_bod*N_blb);
        M_rand.setZero();
        Matrix Dif(3*N_bod*N_blb,3*N_bod*N_blb);
        
        double M_scale = M.norm(); //.lpNorm<2>();
        
        for(int i=0; i<N; ++i){
            if(i % (N/10) == 0){std::cout << "itteration: " << i << "\n";}
            Vector M_half_W1 = M_half_W();
            M_rand += M_half_W1*M_half_W1.transpose();
            Dif = ((1.0/(double) i) * M_rand - M);
            error(i) = (1.0/M_scale)*Dif.norm(); //.lpNorm<2>(); //.lpNorm<Eigen::Infinity>();
        }
        return error;
    }
    
    
    auto RHS_and_Midpoint(RefVector& Slip, RefVector& Force){
        Vector VarKill;
        Vector VarKill_strat;
        double t, elapsed;
        
        std::vector<real_c> r_vecs;
        

        
        if(kBT > 1e-10){
            std::vector<Quat> Qhalf;
            std::vector<Vector> Xhalf;
            Vector M_RFD_vec;
            Vector BI;
            
            while(true){
                // Make Brownian increment for predictor and corrector steps
                t = timeNow();
                Vector M_half_W1 = M_half_W();
                elapsed = timeNow() - t;
                printf( "Lanczos time = %g\n", elapsed );
                Vector M_half_W2;
                if(split_rand){
                    M_half_W2 = M_half_W();
                }
                
                //std::cout << "M*w1: " << M_half_W1.segment<3>(0) << "\n";
                //std::cout << "M*w1: " << M_half_W2.segment<3>(0) << "\n";
                
                // Make M_RFD
                std::cout << "Before RFD\n";
                M_RFD_vec = M_RFD();
                std::cout << "After RFD\n";
                        
                // Set predictor velocity
                double c_1, c_2;
                if(split_rand){
                    c_1 = 2.0*std::sqrt((kBT/dt)); 
                    c_2 = std::sqrt((kBT/dt));
                    BI = c_2*(M_half_W1 - M_half_W2);
                }
                else{
                    c_1 = std::sqrt(2.0*(kBT/dt)); 
                    c_2 = std::sqrt(2.0*(kBT/dt));
                    BI = c_2*(M_half_W1);
                }
                Vector BI_half = c_1*M_half_W1;
                Vector UOm_half = (dt/2.0)*Kinv*BI_half;
                
                std::tie(Qhalf,Xhalf) = update_X_Q(UOm_half);
                r_vecs =  r_vecs_from_cfg(Xhalf,Qhalf);
                
                if(geom == geometry::doubly_periodic){
                    
                    real_c min_z = 1000000.0;
                    real_c max_z = -1000000.0;
                    for (int i = 2; i < r_vecs.size(); i = i + 3) {
                        min_z = std::min(min_z,r_vecs[i]);
                        max_z = std::max(max_z,r_vecs[i]);
                    }
                    
                    if((min_z < pyStar.zmin) || (max_z > pyStar.zmax)){
                        std::cout << "Bad Midpoint!! Repeating step\n";
                        continue;
                    }
                    else{
                        break;
                    }
                }
                else{
                    std::cout << "Not Doubly Periodic\n";
                    break;
                }

                    
                //VarKill = Kill_Variance_Stress(BI);
                //VarKill_strat = Kill_Variance_Strat(BI);
            }
            
            Qs = Qhalf;
            Xs = Xhalf;
            set_K_mats();
            
            // Make RHS for final solve
            Slip += ( (kBT*M_RFD_vec) +  BI); //std::sqrt((2.0*kBT/dt))*M_half_W1 );
        }
        

        Slip *= -1.0;
        Force *= -1.0;
            
        Vector RHS(3*N_bod*N_blb + 6*N_bod);
        RHS << Slip, Force;
        
        //std::cout << RHS << "\n";
        
        return std::make_tuple(RHS,VarKill,VarKill_strat);
  }
  
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Stress Stuff /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

    template<class AVector>
    SparseMd Outer_Product_Mat(AVector& r_vectors, bool transpose){
        
        int size = N_bod*(3*N_blb);  
        
        //Matrix Outer_left(9*N_bod,3*N_bod*N_blb);
        //Matrix Outer_right(9*N_bod,3*N_bod*N_blb);
        //Matrix Trace(9*N_bod,3*N_bod*N_blb);
        //Outer_left.setZero();
        //Outer_right.setZero();
        //Trace.setZero();
        
        std::vector<Trip_d> tripletList;
        
        Vector r_k(3);
        Vector COM_j(3);
        
        
        for(int j = 0; j < N_bod; ++j){
            //std::cout << j << "\n";
            COM_j.setZero();
            /*
            for(int k = 0; k < N_blb; ++k){
                COM_j(0) += (1.0/N_blb)*r_vectors[j*3*N_blb+3*k+0];
                COM_j(1) += (1.0/N_blb)*r_vectors[j*3*N_blb+3*k+1];
                COM_j(2) += (1.0/N_blb)*r_vectors[j*3*N_blb+3*k+2];
            }
            */
            
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
            Rows = 9*N_bod;
            Cols = 3*N_bod*N_blb;
        }
        else{
            Rows = 3*N_bod*N_blb;
            Cols = 9*N_bod;
        }
        SparseMd Outer(Rows,Cols);
        Outer.setFromTriplets(tripletList.begin(), tripletList.end());
        
        
        //std::cout << Outer_left << "\n";
        //std::cout << Outer_right << "\n";
        //Matrix Outer = 0.5*(Outer_right + Outer_left) + Trace;
        //Matrix Outer = Outer_left; // + Trace;
        return Outer;
    }
    
    Vector Apply_Outer_Product(RefVector& Lambda){
        std::vector<real_c> r_vec =  r_vecs_from_cfg(Xs,Qs);
        SparseMd K_S = Outer_Product_Mat(r_vec, false);
        Vector out = K_S*Lambda;
        
        return out;
    }
    
    Vector Stresslet_RFD(){        
        
        double delta = 1.0e-3;
        
        int sz = 6*N_bod;
       // Make random vector
        Vector W = rand_vector(sz);

        
        std::vector<Quat> Qp;
        std::vector<Vector> Xp;
        std::vector<Quat> Qm;
        std::vector<Vector> Xm;
        
        double L_scale = 1.0;
        
        SparseM Kp, Kinvp, Km, Kinvm;
        
        Vector Win = ((delta/2.0)*W);
        for(int d = 0; d < 3; ++d){
            Win[d] *= L_scale;
        }
        
        std::tie(Qp,Xp) = update_X_Q(Win);
        std::vector<real_c> r_vec_p =  r_vecs_from_cfg(Xp,Qp);
        Matrix M_p = rotne_prager_tensor(r_vec_p);
        std::tie(Kp,Kinvp) = Make_K_Kinv(Xp, Qp);
        Matrix M_p_Inv = M_p.completeOrthogonalDecomposition().pseudoInverse();
        Matrix N_p_Inv = Kp.transpose()*M_p_Inv*Kp;
        Matrix N_p = N_p_Inv.completeOrthogonalDecomposition().pseudoInverse();
        SparseMd R_O_p = Outer_Product_Mat(r_vec_p, false);
        //std::cout << Xp[0] << "\n";
        //std::cout << Qp[0].w() << ' ' << Qp[0].x() << ' ' << Qp[0].y() << ' ' << Qp[0].z() << ' ' << "\n";
        //std::cout << "N_p\n";
        //std::cout << N_p << "\n";
        
        Win *= -1.0;
        
        std::tie(Qm,Xm) = update_X_Q(Win);
        std::vector<real_c> r_vec_m =  r_vecs_from_cfg(Xm,Qm);
        Matrix M_m = rotne_prager_tensor(r_vec_m);
        std::tie(Km,Kinvm) = Make_K_Kinv(Xm, Qm);
        Matrix M_m_Inv = M_m.completeOrthogonalDecomposition().pseudoInverse();
        Matrix N_m_Inv = Km.transpose()*M_m_Inv*Km;
        Matrix N_m = N_m_Inv.completeOrthogonalDecomposition().pseudoInverse();
        SparseMd R_O_m = Outer_Product_Mat(r_vec_m, false);
        
        for(int d = 0; d < 3; ++d){
            W[d] *= (1.0/L_scale);
        }
        
        Vector Sp = R_O_p*M_p_Inv*Kp*(N_p*W);
        Vector Sm = R_O_m*M_m_Inv*Km*(N_m*W);
        
        
        Vector out = (kBT/delta)*(Sp-Sm);
                
        return out;
    }
    
    
    Vector K_S_RFD(){        
        double delta = 1.0e-4;
        
        int sz = 3*N_bod*N_blb;
       // Make random vector
        Vector W = rand_vector(sz);
        
        Vector UOM = Kinv*W;
        
        std::vector<Quat> Qp;
        std::vector<Vector> Xp;
        std::vector<Quat> Qm;
        std::vector<Vector> Xm;
        
        
        Vector Win = ((delta/2.0)*UOM);
        std::tie(Qp,Xp) = update_X_Q(Win);
        std::vector<real_c> r_vec_p =  r_vecs_from_cfg(Xp,Qp);
        Win *= -1.0;
        std::tie(Qm,Xm) = update_X_Q(Win);
        std::vector<real_c> r_vec_m =  r_vecs_from_cfg(Xm,Qm);
        
        SparseMd R_O_p = Outer_Product_Mat(r_vec_p, false);
        SparseMd R_O_m = Outer_Product_Mat(r_vec_m, false);
        
        Vector out = (1.0/delta)*((R_O_p*W)-(R_O_m*W));
                
        return out;
    }
    
    
    Vector Kill_Variance_Stress(Vector& Mhalf_W){
        std::vector<real_c> r_vec =  r_vecs_from_cfg(Xs,Qs);
        Matrix M = rotne_prager_tensor(r_vec);
        Matrix M_Inv = M.completeOrthogonalDecomposition().pseudoInverse();
        Matrix N_Inv = KT*M_Inv*K;
        Matrix N = N_Inv.completeOrthogonalDecomposition().pseudoInverse();
        SparseMd R_O = Outer_Product_Mat(r_vec, false);
        Vector Var_Kill = R_O*(M_Inv*Mhalf_W - M_Inv*K*N*KT*(M_Inv*Mhalf_W));
        return Var_Kill;
    }
    
    Vector Kill_Variance_Strat(Vector& Mhalf_W){
        std::vector<real_c> r_vec =  r_vecs_from_cfg(Xs,Qs);
        Matrix M = rotne_prager_tensor(r_vec);
        Matrix M_Inv = M.completeOrthogonalDecomposition().pseudoInverse();
        Matrix N_Inv = KT*M_Inv*K;
        Matrix N = N_Inv.completeOrthogonalDecomposition().pseudoInverse();
        SparseMd R_O = Outer_Product_Mat(r_vec, false);
        Vector Var_Kill = R_O*(M_Inv*K*N*KT*(M_Inv*Mhalf_W));
        return Var_Kill;
    }
    
    Vector Stresslet_Strat(std::vector<real_c>& pos_mid, RefVector& F){ 
        SparseMd K_S = Outer_Product_Mat(pos_mid, false);
        Matrix M_mid = rotne_prager_tensor(pos_mid);
        Matrix M_mid_Inv = M_mid.completeOrthogonalDecomposition().pseudoInverse();
        
        Vector out = K_S*(M_mid_Inv*F);
        
        return out;
    }
    
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// RFD Extra ///////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

    auto Aleks_Terms_RFD(){        
        
        double delta = 1.0e-3;
        
        int sz = 6*N_bod;
        int nb = 3*N_bod*N_blb;
        // Make random vector
        Vector Wv = rand_vector(sz);
        Vector Wx = rand_vector(nb);

        
        std::vector<real_c> r_vec =  r_vecs_from_cfg(Xs,Qs);
        Matrix M = rotne_prager_tensor(r_vec);
        Matrix M_Inv = M.completeOrthogonalDecomposition().pseudoInverse();
        Matrix N_Inv = KT*M_Inv*K;
        Matrix N = N_Inv.completeOrthogonalDecomposition().pseudoInverse();
        SparseMd R_O = Outer_Product_Mat(r_vec, false);
        
        std::vector<Quat> Qp;
        std::vector<Vector> Xp;
        std::vector<Quat> Qm;
        std::vector<Vector> Xm;
        
        double L_scale = 1.0;
        
        SparseM Kp, Kinvp, Km, Kinvm;
        
        Vector Win = ((delta/2.0)*Wv);
        
        std::tie(Qp,Xp) = update_X_Q(Win);
        std::vector<real_c> r_vec_p =  r_vecs_from_cfg(Xp,Qp);
        Matrix M_p = rotne_prager_tensor(r_vec_p);
        std::tie(Kp,Kinvp) = Make_K_Kinv(Xp, Qp);
        Matrix M_p_Inv = M_p.completeOrthogonalDecomposition().pseudoInverse();
        Matrix N_p_Inv = Kp.transpose()*M_p_Inv*Kp;
        Matrix N_p = N_p_Inv.completeOrthogonalDecomposition().pseudoInverse();
        SparseMd R_O_p = Outer_Product_Mat(r_vec_p, false);
        
        Win *= -1.0;
        
        std::tie(Qm,Xm) = update_X_Q(Win);
        std::vector<real_c> r_vec_m =  r_vecs_from_cfg(Xm,Qm);
        Matrix M_m = rotne_prager_tensor(r_vec_m);
        std::tie(Km,Kinvm) = Make_K_Kinv(Xm, Qm);
        Matrix M_m_Inv = M_m.completeOrthogonalDecomposition().pseudoInverse();
        Matrix N_m_Inv = Km.transpose()*M_m_Inv*Km;
        Matrix N_m = N_m_Inv.completeOrthogonalDecomposition().pseudoInverse();
        SparseMd R_O_m = Outer_Product_Mat(r_vec_m, false);
        
        Vector KNWv = K*N*Wv;
        Vector out1 = (1/delta)*(R_O*((M_p_Inv*KNWv)-(M_m_Inv*KNWv)));
        
        
        for (int i=0; i < r_vec.size(); i++) {
            r_vec_p[i] = r_vec[i] + (delta/2.0)*Wx[i];
            r_vec_m[i] = r_vec[i] - (delta/2.0)*Wx[i];
        }
        
        M_p = rotne_prager_tensor(r_vec_p);
        M_m = rotne_prager_tensor(r_vec_m);
        
        Vector divM = (1/delta)*((M_p*Wx)-(M_m*Wx));
        Vector out2 = -1.0*R_O*(M_Inv*divM);
        
        ////////////////////////////////////
        
        Vector NWv = N*Wv;
        Vector out3 = (1/delta)*(R_O_p*(M_p_Inv*(Kp*NWv)) - R_O_m*(M_m_Inv*(Km*NWv)));
        
        Vector dKdN = (1/delta)*((Kp*NWv) - (Km*NWv));
        Vector out4 = R_O*(M_Inv*(dKdN - divM));
        
        Vector MiKNWv = M_Inv*(K*(N*Wv));
        Vector out5  = (1/delta)*(R_O_p*MiKNWv - R_O_m*MiKNWv);
        
        return std::make_tuple(out1,out2,out3,out4,out5);
    }

  
private:

};


using namespace pybind11::literals;
namespace py = pybind11;
//TODO: Fill python documentation here
PYBIND11_MODULE(c_rigid_obj, m) {
    m.doc() = "Rigid code";
    py::class_<CManyBodies>(m, "CManyBodies").
      def(py::init()).
      def("setParameters", &CManyBodies::setParameters,
	  "Set parameters for the module").
      def("setParametersDP", &CManyBodies::setParametersDP,
      "set parameters for DPSTokes module").
      def("setBlkPC", &CManyBodies::setBlkPC,
	  "set PC type").
      def("setWallPC", &CManyBodies::setWallPC,
	  "use wall corrections").
      def("multi_body_pos", &CManyBodies::multi_body_pos,
	  "Get the blob positions").
      def("setConfig", &CManyBodies::setConfig,
	  "Set the X and Q vectors for the current position").
      def("getConfig", &CManyBodies::getConfig,
	  "get the X and Q vectors for the current position").
      def("set_K_mats", &CManyBodies::set_K_mats,
	  "Set the K,K^T,K^-1 matrices for the module").
      def("K_x_U", &CManyBodies::K_x_U,
	  "Multiply K by U").
      def("apply_PC", &CManyBodies::apply_PC<RefVector& >,
	  "apply for PC").
      def("apply_Saddle", &CManyBodies::apply_Saddle<RefVector& >,
	  "apply for [M, -K;-K^T, 0]").
      def("test_PC", &CManyBodies::test_PC<RefVector& >,
	  "test_PC").
      def("Test_Mhalf", &CManyBodies::Test_Mhalf,
	  "Test_Mhalf").
      def("apply_M",&CManyBodies::apply_M<RefVector& >,
      "apply M").
      def("update_X_Q_out", &CManyBodies::update_X_Q_out,
	  "update_X_Q_out").
      def("KTinv_RFD", &CManyBodies::KTinv_RFD,
	  "KTinv_RFD").
      def("M_RFD", &CManyBodies::M_RFD,
	  "M_RFD").
      def("Stresslet_RFD", &CManyBodies::Stresslet_RFD,
	  "Stresslet_RFD").
      def("Apply_Outer_Product", &CManyBodies::Apply_Outer_Product,
	  "Apply_Outer_Product").
      def("Stresslet_Strat", &CManyBodies::Stresslet_Strat,
	  "Stresslet_Strat").
      def("K_S_RFD", &CManyBodies::K_S_RFD,
	  "K_S_RFD").
      def("Aleks_Terms_RFD", &CManyBodies::Aleks_Terms_RFD,
	  "Aleks_Terms_RFD").
      def("RHS_and_Midpoint", &CManyBodies::RHS_and_Midpoint,
	  "RHS_and_Midpoint").
      def("evolve_X_Q", &CManyBodies::evolve_X_Q,
	  "evolve_X_Q");
}
