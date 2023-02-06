#include <iostream>
#include <fstream>
#include <utility> 
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

// double typedefs
typedef Eigen::MatrixXd Matrix;

// rigid types
typedef Eigen::Vector3d Vector3;
typedef Eigen::Matrix3d Matrix3;




class C_Lubrication
{
  private:
  void SetMemberData(std::string fname, std::vector< std::vector<double> >& vec_11, 
        std::vector< std::vector<double> >& vec_12, std::vector<double>& x);
  void SetMemberDataWall(std::string fname, std::vector< std::vector<double> >& vec, std::vector<double>& x, bool reverse);
  std::vector< std::vector<double> > mob_scalars_WS_11, mob_scalars_WS_12;
  std::vector<double> WS_x;
  std::vector< std::vector<double> > mob_scalars_JO_11, mob_scalars_JO_12;
  std::vector<double> JO_x;
  std::vector< std::vector<double> > mob_scalars_MB_11, mob_scalars_MB_12;
  std::vector<double> MB_x;
  std::vector< std::vector<double> > mob_scalars_wall_2562;
  std::vector<double> Wall_2562_x;
  std::vector< std::vector<double> > mob_scalars_wall_MB;
  std::vector<double> Wall_MB_x;
  int FindNearestIndexLower(double r_norm, std::vector<double>& x);
  double LinearInterp(double r_norm, double xL, double xR, double yL, double yR);
  void ResistMatrix(double r_norm, double mob_factor[3], Vector3 r_hat, Matrix& R, bool inv, std::vector<double>& x, 
        const std::vector< std::vector<double> >& vec_11, const std::vector< std::vector<double> >& vec_12);
  void ATResistMatrix(double r_norm, double mob_factor[3], Vector3 r_hat, Matrix& R);
  Matrix WallResistMatrix(double r_norm, double mob_factor[3], std::vector< double >& x, 
          const std::vector< std::vector< double > >& vec);
  Matrix  ResistPairLub(double r_norm, double mob_factor[3], Vector3 r_hat);
  Matrix WallResistMatrixBlob(double r_norm, double mob_factor[3], std::vector< double >& x, 
          const std::vector< std::vector< double > >& vec);
  Matrix ResistPairBlob(double r_norm, double mob_factor[3], Vector3 r_hat);

  std::vector<Eigen::Triplet<double>> makeTriplets(const std::vector<double>& data, 
                                    const std::vector<int>& rows, const std::vector<int>& cols);

  double tolerance = 10e-4;
  double cutoff_wall = 1.0e10;

  public:
  void Set_R_Mats(py::list r_vecs, py::list neighbors, double a, double eta, double cutoff, double cutoff_wall, 
        std::vector<double> periodic_length, bool Sup_if_true);
  void ResistPairLub_py(double r_norm, double a, double eta, py::array_t<double> r_hat);
  void ResistCOO(py::list r_vectors, py::list n_list, double a, double eta, double cutoff, double wall_cutoff, 
        std::vector<double> periodic_length, bool Sup_if_true, std::vector<double>& data, std::vector<int>& rows, 
        std::vector<int>& cols);
  void ResistCOO_wall(py::list r_vectors, double a, double eta, double wall_cutoff, std::vector<double> periodic_length, 
        bool Sup_if_true, std::vector<double>& data, std::vector<int>& rows, std::vector<int>& cols);
  double debye_cut;
  C_Lubrication(double d_cut);
  Eigen::SparseMatrix<double> lub_print();

  Eigen::SparseMatrix<double> R_blob;
  Eigen::SparseMatrix<double> R_lub;
  Eigen::SparseMatrix<double> Delta_R;
};

C_Lubrication::C_Lubrication(double d_cut)
{
  Py_Initialize();
  debye_cut = d_cut;
  std::string base_dir = __FILENAME__;
//   std::cout << "++++++++++++++++++++++++++++\n";
//   std::cout << base_dir << "\n";
//   std::cout << base_dir+"/Resistance_Coefs/mob_scalars_WS.txt" << "\n";
//   std::cout << "++++++++++++++++++++++++++++\n";
  SetMemberData(base_dir+"/Resistance_Coefs/mob_scalars_WS.txt",mob_scalars_WS_11,mob_scalars_WS_12,WS_x);
  SetMemberData(base_dir+"/Resistance_Coefs/res_scalars_JO.txt",mob_scalars_JO_11,mob_scalars_JO_12,JO_x);
  SetMemberDataWall(base_dir+"/Resistance_Coefs/mob_scalars_wall_MB_2562_eig_thresh.txt",mob_scalars_wall_2562,Wall_2562_x,true);
  SetMemberData(base_dir+"/Resistance_Coefs/res_scalars_MB_1.txt",mob_scalars_MB_11,mob_scalars_MB_12,MB_x);
  SetMemberDataWall(base_dir+"/Resistance_Coefs/res_scalars_wall_MB.txt",mob_scalars_wall_MB,Wall_MB_x,false);
}

void C_Lubrication::SetMemberData(std::string fname, std::vector< std::vector<double> >& vec_11, 
                    std::vector< std::vector<double> >& vec_12, std::vector<double>& x)
{ 
  std::ifstream ifs(fname);
  double tempval;
  std::vector<double> tempv;
  
  if (!ifs.fail())
  {
    int p = 0;
    int c = -1;
    while(!ifs.eof())
    {
      c++;
      ifs >> tempval;
      tempv.push_back(tempval);
      if(c == 5)
      {
	p++;
	c=-1;
	if(p % 2){
	  vec_11.push_back(tempv);
	}
	else{
	  vec_12.push_back(tempv);
	}
	tempv.clear();
      }
    }
    ifs.close();
  }
  
  int k = 0;
  for (auto row : vec_11) {
    k++;
    x.push_back(row[0]);
  }

}


void C_Lubrication::SetMemberDataWall(std::string fname, std::vector< std::vector<double> >& vec, 
                    std::vector<double>& x, bool reverse)
{ 
  std::ifstream ifs(fname);
  double tempval;
  std::vector<double> tempv;
  
  if (!ifs.fail())
  {
    int c = -1;
    while(!ifs.eof())
    {
      c++;
      ifs >> tempval;
      tempv.push_back(tempval);
      if(c == 5)
      {
	c=-1;
	if(reverse)
	{
	  vec.insert(vec.begin(), tempv);
	}
	else
	{
	  vec.push_back(tempv);
	}
	tempv.clear();
      }
    }
    ifs.close();
  }
  
  int k = 0;
  for (auto row : vec) {
    k++;
    x.push_back(row[0]);
  }

}

int C_Lubrication::FindNearestIndexLower(double r_norm, std::vector<double>& x)
{
    // TODO: should make x a const vector but then distance fails
    // dunno what to do 
    std::vector<double>::iterator before, after, it;
    before = std::lower_bound(x.begin(), x.end(), r_norm);
    if(before == x.begin()){return -1;}
    if(before == x.end()){return x.size()-1;}
    
    after = before;
    --before;

    return std::distance(x.begin(),before);
    
}

double C_Lubrication::LinearInterp(double r_norm, double xL, double xR, double yL, double yR)
{
    if(r_norm < xL || r_norm > xR){std::cout << "error in linear interp." << std::endl; return 1e100;}
    double dydx = ( yR - yL ) / ( xR - xL ); 
    return yL + dydx * ( r_norm - xL ); 
}

void C_Lubrication::ResistMatrix(double r_norm, double mob_factor[3], Vector3 r_hat, Matrix& R, bool inv, 
                  std::vector< double >& x, const std::vector< std::vector< double > >& vec_11, 
                  const std::vector< std::vector< double > >& vec_12)
{
  
    double X11A, Y11A, Y11B, X11C, Y11C; 
    double X12A, Y12A, Y12B, X12C, Y12C;
    
    int Ind = FindNearestIndexLower(r_norm, x);
    
    if(Ind == -1 || Ind == x.size()-1)
    {
      int edge = (Ind == -1) ? 0 : (x.size()-1);
      X11A = vec_11[edge][1];
      Y11A = vec_11[edge][2];
      Y11B = vec_11[edge][3];
      X11C = vec_11[edge][4];
      Y11C = vec_11[edge][5]; 
      
      X12A = vec_12[edge][1]; 
      Y12A = vec_12[edge][2];
      Y12B = vec_12[edge][3];
      X12C = vec_12[edge][4];
      Y12C = vec_12[edge][5];
    }
    else
    {
      double a_11[5], a_12[5];
      double xL, yL11, yL12, xR, yR11, yR12;
      
      for(int i = 0; i < 5; i++){
	xL = x[Ind]; xR = x[Ind+1];
	yL11 = vec_11[Ind][i+1], yR11 = vec_11[Ind+1][i+1];
	yL12 = vec_12[Ind][i+1], yR12 = vec_12[Ind+1][i+1];
	a_11[i] = LinearInterp(r_norm, xL, xR, yL11, yR11);
	a_12[i] = LinearInterp(r_norm, xL, xR, yL12, yR12);
      }
      
      X11A = a_11[0];
      Y11A = a_11[1];
      Y11B = a_11[2];
      X11C = a_11[3];
      Y11C = a_11[4]; 
      
      X12A = a_12[0]; 
      Y12A = a_12[1];
      Y12B = a_12[2];
      X12C = a_12[3];
      Y12C = a_12[4];
    }
    
    Matrix3 squeezeMat = r_hat * r_hat.transpose();
    Matrix3 Eye;
    Eye.setIdentity(3,3);
    Matrix3 shearMat = Eye - squeezeMat;
    Matrix3 vortMat;
    vortMat << 0.0, r_hat[2], -r_hat[1],
               -r_hat[2], 0.0, r_hat[0],
               r_hat[1], -r_hat[0], 0.0;

    vortMat *= -1;
	       
	       
	       
    R.block<3,3>(0,0) = mob_factor[0]*(X11A*squeezeMat + Y11A*shearMat);
    R.block<3,3>(0,3) = -mob_factor[1]*(Y11B*vortMat); // neg
    R.block<3,3>(0,6) = mob_factor[0]*(X12A*squeezeMat + Y12A*shearMat);
    R.block<3,3>(0,9) = mob_factor[1]*(Y12B*vortMat); 
    
    R.block<3,3>(3,0) = mob_factor[1]*(Y11B*vortMat);
    R.block<3,3>(3,3) = mob_factor[2]*(X11C*squeezeMat + Y11C*shearMat);
    R.block<3,3>(3,6) = mob_factor[1]*(Y12B*vortMat);
    R.block<3,3>(3,9) = mob_factor[2]*(X12C*squeezeMat + Y12C*shearMat);
    
    R.block<3,3>(6,0) = mob_factor[0]*(X12A*squeezeMat + Y12A*shearMat);
    R.block<3,3>(6,3) = -mob_factor[1]*(Y12B*vortMat); // neg
    R.block<3,3>(6,6) = mob_factor[0]*(X11A*squeezeMat + Y11A*shearMat);
    R.block<3,3>(6,9) = mob_factor[1]*(Y11B*vortMat);
    
    R.block<3,3>(9,0) = -mob_factor[1]*(Y12B*vortMat); // neg
    R.block<3,3>(9,3) = mob_factor[2]*(X12C*squeezeMat + Y12C*shearMat);
    R.block<3,3>(9,6) = -mob_factor[1]*(Y11B*vortMat); // neg
    R.block<3,3>(9,9) = mob_factor[2]*(X11C*squeezeMat + Y11C*shearMat);
    
    if(inv){
      R = R.inverse();
    }
    
}

void C_Lubrication::ATResistMatrix(double r_norm, double mob_factor[3], Vector3 r_hat, Matrix& R)
{
  
    double X11A, Y11A, Y11B, X11C, Y11C; 
    double X12A, Y12A, Y12B, X12C, Y12C;
    
    double epsilon = r_norm-2.0;
      
    X11A = 0.995419E0+(0.25E0)*(1.0/epsilon)+(0.225E0)*log((1.0/epsilon))+(0.267857E-1)*epsilon*log((1.0/epsilon));
    X12A = (-0.350153E0)+(-0.25E0)*(1.0/epsilon)+(-0.225E0)*log((1.0/epsilon))+(-0.267857E-1)*epsilon*log((1.0/epsilon));
    Y11A = 0.998317E0+(0.166667E0)*log((1.0/epsilon));
    Y12A = (-0.273652E0)+(-0.166667E0)*log((1.0/epsilon));
    Y11B = (-0.666667E0)*(0.23892E0+(-0.25E0)*log((1.0/epsilon))+(-0.125E0)*epsilon*log((1.0/epsilon)));
    Y12B = (-0.666667E0)*((-0.162268E-2)+(0.25E0)*log((1.0/epsilon))+(0.125E0)*epsilon*log((1.0/epsilon)));
    X11C = 0.133333E1*(0.10518E1+(-0.125E0)*epsilon*log((1.0/epsilon)));
    X12C = 0.133333E1*((-0.150257E0)+(0.125E0)*epsilon*log((1.0/epsilon)));
    Y11C = 0.133333E1*(0.702834E0+(0.2E0)*log((1.0/epsilon))+(0.188E0)*epsilon*log((1.0/epsilon)));
    Y12C = 0.133333E1*((-0.27464E-1)+(0.5E-1)*log((1.0/epsilon))+(0.62E-1)*epsilon*log((1.0/epsilon)));
    
    Matrix3 squeezeMat = r_hat * r_hat.transpose();
    Matrix3 Eye;
    Eye.setIdentity(3,3);
    Matrix3 shearMat = Eye - squeezeMat;
    Matrix3 vortMat;
    vortMat << 0.0, r_hat[2], -r_hat[1],
               -r_hat[2], 0.0, r_hat[0],
               r_hat[1], -r_hat[0], 0.0;

    vortMat *= -1;
	       
	       
	       
    R.block<3,3>(0,0) = mob_factor[0]*(X11A*squeezeMat + Y11A*shearMat);
    R.block<3,3>(0,3) = -mob_factor[1]*(Y11B*vortMat); // neg
    R.block<3,3>(0,6) = mob_factor[0]*(X12A*squeezeMat + Y12A*shearMat);
    R.block<3,3>(0,9) = mob_factor[1]*(Y12B*vortMat); 
    
    R.block<3,3>(3,0) = mob_factor[1]*(Y11B*vortMat);
    R.block<3,3>(3,3) = mob_factor[2]*(X11C*squeezeMat + Y11C*shearMat);
    R.block<3,3>(3,6) = mob_factor[1]*(Y12B*vortMat);
    R.block<3,3>(3,9) = mob_factor[2]*(X12C*squeezeMat + Y12C*shearMat);
    
    R.block<3,3>(6,0) = mob_factor[0]*(X12A*squeezeMat + Y12A*shearMat);
    R.block<3,3>(6,3) = -mob_factor[1]*(Y12B*vortMat); // neg
    R.block<3,3>(6,6) = mob_factor[0]*(X11A*squeezeMat + Y11A*shearMat);
    R.block<3,3>(6,9) = mob_factor[1]*(Y11B*vortMat);
    
    R.block<3,3>(9,0) = -mob_factor[1]*(Y12B*vortMat); // neg
    R.block<3,3>(9,3) = mob_factor[2]*(X12C*squeezeMat + Y12C*shearMat);
    R.block<3,3>(9,6) = -mob_factor[1]*(Y11B*vortMat); // neg
    R.block<3,3>(9,9) = mob_factor[2]*(X11C*squeezeMat + Y11C*shearMat);
    
}

Matrix C_Lubrication::WallResistMatrix(double r_norm, double mob_factor[3], std::vector< double >& x, 
                      const std::vector< std::vector< double > >& vec)
{
  
    double Xa, Ya, Yb, Xc, Yc; 
    double Xa_asym, Ya_asym, Yb_asym, Xc_asym, Yc_asym; 
    double Xa_cutoff, Ya_cutoff, Yb_cutoff, Xc_cutoff, Yc_cutoff; 
    double RXa, RYa, RYb, RXc, RYc;
    double epsilon = r_norm-1.0;
    
    double tanh_fact = 1.0;
    if(epsilon < debye_cut) // epsilon < 0.1*debye_cut
    {
      epsilon = debye_cut;
      //tanh_fact = (1.0+tanh((epsilon+5*debye_cut)/2/debye_cut))*0.5;
      //epsilon = fmin(fabs(epsilon),debye_cut); //debye_cut;
      //if(epsilon < 0.1*debye_cut){epsilon = 0.1*debye_cut;}
      r_norm = 1.0+epsilon; //1.0+debye_cut;
    }
      
    int Ind = FindNearestIndexLower(r_norm, x);
    if(Ind == -1)
    {
      Xa = vec[0][1];
      Ya = vec[0][2];
      Yb = vec[0][3];
      Xc = vec[0][4];
      Yc = vec[0][5]; 
    }
    else if(Ind == x.size()-1)
    {
      Xa = 1.0 - (9.0/8.0)*(1.0/r_norm);
      Ya = 1.0 - (9.0/16.0)*(1.0/r_norm);
      Yb = 0.0;
      Xc = 0.75;
      Yc = 0.75;
    }
    else
    {
      double a[5];
      double xL, yL, xR, yR;
      
      for(int i = 0; i < 5; i++){
	xL = x[Ind]; xR = x[Ind+1];
	yL = vec[Ind][i+1], yR = vec[Ind+1][i+1];
	a[i] = LinearInterp(r_norm, xL, xR, yL, yR);
      }
      
      Xa = a[0];
      Ya = a[1];
      Yb = a[2];
      Xc = a[3];
      Yc = a[4]; 
    }
    
    
    
    Xa_asym = 1.0/epsilon - (1.0/5.0)*log(epsilon) + 0.971280;
    Ya_asym = -(8.0/15.0)*log(epsilon) + 0.9588;
    Yb_asym = -(-(1.0/10.0)*log(epsilon)-0.1895) - 0.4576*epsilon;
    Yb_asym *= 4./3.;
    Xc_asym = 1.2020569 - 3.0*(M_PI*M_PI/6.0-1.0)*epsilon;
    Xc_asym *= 4./3.;
    Yc_asym = -2.0/5.0*log(epsilon) + 0.3817 + 1.4578*epsilon;
    Yc_asym *= 4./3.;
    
    Xa_cutoff = 1.+0.1;
    Ya_cutoff = 1.+0.01;
    Yb_cutoff = 1.+0.1;
    Xc_cutoff = 1.+0.01;
    Yc_cutoff = 1.+0.1;
    
    double denom = Ya*Yc - Yb*Yb;
    RXa = 1.0/Xa;
    RYa = Yc/denom;
    RYb = -Yb/denom;
    RXc = 1.0/Xc;
    RYc = Ya/denom; 
    
    Xa = (r_norm > Xa_cutoff) ? RXa : Xa_asym;
    Ya = (r_norm > Ya_cutoff) ? RYa : Ya_asym;
    Yb = (r_norm > Yb_cutoff) ? RYb : Yb_asym;
    Xc = (r_norm > Xc_cutoff) ? RXc : Xc_asym;
    Yc = (r_norm > Yc_cutoff) ? RYc : Yc_asym;
    
    double XcPlus = fmax((Xc-4.0/3.0),0.0);
    double YcPlus = fmax((Yc-4.0/3.0),0.0);
    
    Matrix R(6,6);
    R << mob_factor[0]*(Ya-1.), 0, 0, 0, mob_factor[1]*Yb, 0,
	 0, mob_factor[0]*(Ya-1.), 0, -mob_factor[1]*Yb, 0, 0,
	 0, 0, mob_factor[0]*(Xa-1.), 0, 0, 0,
	 0, -mob_factor[1]*Yb, 0, mob_factor[2]*YcPlus, 0, 0,
	 mob_factor[1]*Yb, 0, 0, 0, mob_factor[2]*YcPlus, 0,
	 0, 0, 0, 0, 0, mob_factor[2]*XcPlus;
         

    if(fabs(tanh_fact-1.0) > 1e-10)
    {
      R *= tanh_fact;
    }
	 
    return R;
}


Matrix C_Lubrication::WallResistMatrixBlob(double r_norm, double mob_factor[3], std::vector< double >& x, 
                      const std::vector< std::vector< double > >& vec)
{
  
    double Xa, Ya, Yb, Xc, Yc; 
    
    double epsilon = r_norm-1.0;
    double tanh_fact = 1.0;
    if(epsilon < debye_cut) // epsilon < 0.1*debye_cut
    {
      epsilon = debye_cut;
      //tanh_fact = (1.0+tanh((epsilon+5*debye_cut)/2/debye_cut))*0.5;
      //epsilon = fmin(fabs(epsilon),debye_cut); //debye_cut;
      //if(epsilon < 0.1*debye_cut){epsilon = 0.1*debye_cut;}
      r_norm = 1.0+epsilon; //1.0+debye_cut;
    }
      
    int Ind = FindNearestIndexLower(r_norm, x);
    if(Ind == -1)
    {
      Xa = vec[0][1];
      Ya = vec[0][2];
      Yb = vec[0][3];
      Xc = vec[0][4];
      Yc = vec[0][5]; 
    }
    else if(Ind == x.size()-1)
    {
      Xa = 1.0/(1.0 - (9.0/8.0)*(1.0/r_norm));
      Ya = 1.0/(1.0 - (9.0/16.0)*(1.0/r_norm));
      Yb = 0.0;
      Xc = 1.0/0.75;
      Yc = 1.0/0.75;
    }
    else
    {
      double a[5];
      double xL, yL, xR, yR;
      
      for(int i = 0; i < 5; i++){
	xL = x[Ind]; xR = x[Ind+1];
	yL = vec[Ind][i+1], yR = vec[Ind+1][i+1];
	a[i] = LinearInterp(r_norm, xL, xR, yL, yR);
      }
      
      Xa = a[0];
      Ya = a[1];
      Yb = a[2];
      Xc = a[3];
      Yc = a[4]; 
    }
    
    Matrix R(6,6);
    R << mob_factor[0]*(Ya-1.), 0, 0, 0, mob_factor[1]*Yb, 0,
	 0, mob_factor[0]*(Ya-1.), 0, -mob_factor[1]*Yb, 0, 0,
	 0, 0, mob_factor[0]*(Xa-1.), 0, 0, 0,
	 0, -mob_factor[1]*Yb, 0, mob_factor[2]*(Yc-4.0/3.0), 0, 0,
	 mob_factor[1]*Yb, 0, 0, 0, mob_factor[2]*(Yc-4.0/3.0), 0,
	 0, 0, 0, 0, 0, mob_factor[2]*(Xc-4.0/3.0);
         

    if(fabs(tanh_fact-1.0) > 1e-10)
    {
      R *= tanh_fact;
    }
	 
    return R;
}


Matrix C_Lubrication::ResistPairLub(double r_norm, double mob_factor[3], Vector3 r_hat)
{
    double AT_cutoff = (2+0.006-1e-8);
    double WS_cutoff = (2+0.1+1e-8);
    bool inv;
    double res_factor[3] = {1.0/mob_factor[0], 1.0/mob_factor[1], 1.0/mob_factor[2]};
    Matrix R(12,12);
    
    double epsilon = r_norm-2.0;
    double tanh_fact = 1.0;
    if(epsilon < debye_cut) // epsilon < 0.1*debye_cut
    {
      epsilon = debye_cut;
      //tanh_fact = (1.0+tanh((epsilon+5*debye_cut)/2/debye_cut))*0.5;
      //epsilon = fmin(fabs(epsilon),debye_cut); //debye_cut;
      //if(epsilon < 0.1*debye_cut){epsilon = 0.1*debye_cut;}
      r_norm = epsilon+2.0; //r_norm = 2.0+debye_cut;
    }
    
    if(r_norm <= AT_cutoff)
    {
      //std::cout << "AT being used \n";
      ATResistMatrix(r_norm, mob_factor, r_hat, R);
    }
    else if(r_norm <= WS_cutoff)
    {
      inv=true;
      ResistMatrix(r_norm, res_factor, r_hat, R, inv, WS_x, mob_scalars_WS_11, mob_scalars_WS_12);
    }
    else
    {
      inv = false;
      ResistMatrix(r_norm, mob_factor, r_hat, R, inv, JO_x, mob_scalars_JO_11, mob_scalars_JO_12);
    }
    
    if(fabs(tanh_fact-1.0) > 1e-10)
    {
      R *= tanh_fact;
    }

    return R;
}

Matrix C_Lubrication::ResistPairBlob(double r_norm, double mob_factor[3], Vector3 r_hat)
{
    bool inv=false;
    Matrix R(12,12);
    
    double epsilon = r_norm-2.0;
    double tanh_fact = 1.0;
    if(epsilon < debye_cut) // epsilon < 0.1*debye_cut
    {
      epsilon = debye_cut;
      //tanh_fact = (1.0+tanh((epsilon+5*debye_cut)/2/debye_cut))*0.5;
      //////////
      //epsilon = fmin(fabs(epsilon),debye_cut); //debye_cut;
      //if(epsilon < 0.1*debye_cut){epsilon = 0.1*debye_cut;}
      r_norm = epsilon+2.0; //2.0+debye_cut;
    }
    
    ResistMatrix(r_norm, mob_factor, r_hat, R, inv, MB_x, mob_scalars_MB_11, mob_scalars_MB_12);

    if(fabs(tanh_fact-1.0) > 1e-10)
    {
      R *= tanh_fact;
    }
    
    return R;
}

// TODO remove, just for testing?
void C_Lubrication::ResistPairLub_py(double r_norm, double a, double eta, py::array_t<double> r_hat)
{
    Vector3 r_hat_E; 
    r_hat_E << r_hat.at(0), r_hat.at(1), r_hat.at(2);
    double mob_factor[3] = {(6.0*M_PI*eta*a), (6.0*M_PI*eta*a*a), (6.0*M_PI*eta*a*a*a)};
    Matrix R = ResistPairLub(r_norm, mob_factor, r_hat_E);
        {std::cout << "[" << mob_factor[0] << " " << mob_factor[1] << " " << mob_factor[2] << "]" << std::endl;}
	{std::cout << "[" << r_hat_E[0] << " " << r_hat_E[1] << " " << r_hat_E[2] << "]" << std::endl;}
	{std::cout << r_norm << "\n"; std::cout << R << std::endl; }
}

void C_Lubrication::ResistCOO(py::list r_vectors, py::list n_list, double a, double eta, double cutoff, 
                             double wall_cutoff, std::vector<double> periodic_length, bool Sup_if_true, 
                             std::vector<double>& data, std::vector<int>& rows, std::vector<int>& cols)
{
  int num_bodies = r_vectors.size();
  double mob_factor[3] = {(6.0*M_PI*eta*a), (6.0*M_PI*eta*a*a), (6.0*M_PI*eta*a*a*a)};
  std::vector<double> L = periodic_length;
  int k, num_neighbors;
  Vector3 r_jk, r_hat;
  double r_norm, height;
  Matrix R_pair, R_wall, R_pair_jj, R_pair_kk, R_pair_kj, R_pair_jk;
  double R_val;
  double m_eps = 1e-12;
  
  for(int j = 0; j < num_bodies; j++)
  {
    py::array_t<double> r_j = py::cast<py::array_t<double>>(r_vectors[j]);
    height = r_j.at(2);
    height /= a;
    
      if(height < wall_cutoff)
      {
	if(Sup_if_true)
	{
	  R_wall = WallResistMatrix(height, mob_factor, Wall_2562_x, mob_scalars_wall_2562);
	}
	else
	{
	  R_wall = WallResistMatrixBlob(height, mob_factor, Wall_MB_x, mob_scalars_wall_MB);
	}

	for(int row = 0; row < 6; row++)
	{
	  for(int col = 0; col < 6; col++)
	  {
	    R_val = R_wall(row,col);
	    if(fabs(R_val) > m_eps)
	    {
	      data.push_back(R_val);
	      rows.push_back(row+j*6);
	      cols.push_back(col+j*6);
	    }
	  } // col  
	} // row
      }// if wall_cutoff
      
      py::array_t<int> neighbors = py::cast<py::array_t<int>>(n_list[j]);
      num_neighbors = neighbors.size();
      if(num_neighbors == 0){continue;}
      
      for(int k_ind = 0; k_ind < neighbors.size(); k_ind++)
      {
	k = neighbors.at(k_ind);
	
	py::array_t<double> r_k = py::cast<py::array_t<double>>(r_vectors[k]);
	for(int l = 0; l < 3; ++l)
	{
	  r_jk[l] = (r_j.at(l) - r_k.at(l));
	  if(L.at(l) > 0)
	  {
	    r_jk[l] = r_jk[l] - int(r_jk[l] / L.at(l) + 0.5 * (int(r_jk[l]>0) - int(r_jk[l]<0))) * L.at(l);
	    r_jk[l] *= (1./a);
	  }
	}
	r_norm = r_jk.norm();
	r_hat = -r_jk/r_norm; ////// NEGATIVE SIGN
	
	if(r_norm < cutoff)
	{
	  if(Sup_if_true)
	  {
	    R_pair = ResistPairLub(r_norm, mob_factor,r_hat);
	  }
	  else
	  {
	    R_pair = ResistPairBlob(r_norm, mob_factor,r_hat);
	  }
	  R_pair_jj = R_pair.block<6,6>(0,0);
	  R_pair_kk = R_pair.block<6,6>(6,6);
	  R_pair_jk = R_pair.block<6,6>(0,6);
	  R_pair_kj = R_pair.block<6,6>(6,0);
	  
  // 	if(j == 0){std::cout << "[" << mob_factor[0] << " " << mob_factor[1] << " " << mob_factor[2] << "]" << std::endl;}
  // 	if(j == 0){std::cout << "[" << r_hat[0] << " " << r_hat[1] << " " << r_hat[2] << "]" << std::endl;}
  // 	if(j == 0){std::cout << r_norm << "\n"; std::cout << std::setprecision(5) << R_pair_jj << std::endl; }
	  
	  
	  for(int row = 0; row < 6; row++)
	  {
	    for(int col = 0; col < 6; col++)
	    {
	      // jj block
	      R_val = R_pair_jj(row,col);
	      if(fabs(R_val) > m_eps)
	      {
		data.push_back(R_val);
		rows.push_back(row+j*6);
		cols.push_back(col+j*6);
	      }
	      
	      // kk block
	      R_val = R_pair_kk(row,col);
	      if(fabs(R_val) > m_eps)
	      {
		data.push_back(R_val);
		rows.push_back(row+k*6);
		cols.push_back(col+k*6);
	      }
	      
	      // jk block
	      R_val = R_pair_jk(row,col);
	      if(fabs(R_val) > m_eps)
	      {
		data.push_back(R_val);
		rows.push_back(row+j*6);
		cols.push_back(col+k*6);
	      }
	      
	      // kj block
	      R_val = R_pair_kj(row,col);
	      if(fabs(R_val) > m_eps)
	      {
		data.push_back(R_val);
		rows.push_back(row+k*6);
		cols.push_back(col+j*6);
	      }
	      
	    } // cols
	  } // rows
	
	} // if r < cutoff
	
      }// loop over k
    
  } // loop over j
  
}

void C_Lubrication::ResistCOO_wall(py::list r_vectors, double a, double eta, double wall_cutoff, 
                    std::vector<double> periodic_length, bool Sup_if_true, std::vector<double>& data, 
                    std::vector<int>& rows, std::vector<int>& cols)
{
  int num_bodies = r_vectors.size();
  double mob_factor[3] = {(6.0*M_PI*eta*a), (6.0*M_PI*eta*a*a), (6.0*M_PI*eta*a*a*a)};
  std::vector<double> L = periodic_length;
  double r_norm, height;
  Matrix R_wall;
  double R_val;
  double m_eps = 1e-12;
  
  for(int j = 0; j < num_bodies; j++)
  {
    py::array_t<double> r_j = py::cast<py::array_t<double>>(r_vectors[j]);
    
    height = r_j.at(2);
    height /= a;
    
    if(height < wall_cutoff){continue;}
    
    if(Sup_if_true)
    {
      R_wall = WallResistMatrix(height, mob_factor, Wall_2562_x, mob_scalars_wall_2562);
    }
    else
    {
      R_wall = WallResistMatrixBlob(height, mob_factor, Wall_MB_x, mob_scalars_wall_MB);
    }

    for(int row = 0; row < 6; row++)
    {
      for(int col = 0; col < 6; col++)
      {
	R_val = R_wall(row,col);
	if(fabs(R_val) > m_eps)
	{
	  data.push_back(R_val);
	  rows.push_back(row+j*6);
	  cols.push_back(col+j*6);
	}
      } // col
    } // row 
  } // j loop
}

void C_Lubrication::Set_R_Mats(py::list r_vecs, py::list neighbors, double a, double eta, double cutoff, 
                              double cutoff_wall, std::vector<double> periodic_length, bool Sup_if_true){

  int num_particles = r_vecs.size();

  double small = 0.5*6.0*M_PI*eta*a*tolerance;

  std::vector<double> data;
  std::vector<int> rows, cols;

  ResistCOO(r_vecs, neighbors, a, eta, cutoff, cutoff_wall, periodic_length, false, data, rows, cols);

  Eigen::SparseMatrix<double> R_blob_cut(6*num_particles, 6*num_particles);
  if(data.size() > 0){
    std::vector<Eigen::Triplet<double>> triplets = makeTriplets(data, rows, cols);
    R_blob_cut.setFromTriplets(triplets.begin(), triplets.end());

  } else{
    // inserts a diag
    // said to be fast creation method here: https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
    for(int i = 0; i < 6*num_particles; i++){
      R_blob_cut.insert(i, i) = small;
    }
  }


  std::cout << "------------------- C++ --------------" << std::endl;

  std::cout << "C++ R BLOB CUT NNZ: " << R_blob_cut.nonZeros() << std::endl;

  data.clear();
  rows.clear();
  cols.clear();

  ResistCOO_wall(r_vecs, a, eta, cutoff, periodic_length, false, data, rows, cols);

  Eigen::SparseMatrix<double> R_blob_cut_wall(6*num_particles, 6*num_particles);
  if(data.size() > 0){
    std::vector<Eigen::Triplet<double>> triplets = makeTriplets(data, rows, cols);
    R_blob_cut_wall.setFromTriplets(triplets.begin(), triplets.end());

  } else{
    for(int i = 0; i < 6*num_particles; i++){
      R_blob_cut_wall.insert(i, i) = small;
    }
  }

  std::cout << "C++ R BLOB CUT WALL NNZ: " << R_blob_cut_wall.nonZeros() << std::endl;
  
  C_Lubrication::R_blob = R_blob_cut + R_blob_cut_wall;
  std::cout << Matrix(R_blob) << std::endl;
  std::cout << "R BLOB IS COMPRESSED " << R_blob.isCompressed() <<  std::endl;


  data.clear();
  data.clear();
  data.clear();

  ResistCOO(r_vecs, neighbors, a, eta, cutoff, cutoff_wall, periodic_length, true, data, rows, cols);

  Eigen::SparseMatrix<double> R_lub_cut(6*num_particles, 6*num_particles);
  if(data.size() > 0){
    std::vector<Eigen::Triplet<double>> triplets = makeTriplets(data, rows, cols);
    R_lub_cut.setFromTriplets(triplets.begin(), triplets.end());

  } else{
    // inserts a diag
    // said to be fast creation method here: https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
    for(int i = 0; i < 6*num_particles; i++){
      R_lub_cut.insert(i, i) = small;
    }
  }

  data.clear();
  rows.clear();
  cols.clear();

  ResistCOO_wall(r_vecs, a, eta, cutoff, periodic_length, true, data, rows, cols);

  Eigen::SparseMatrix<double> R_lub_cut_wall(6*num_particles, 6*num_particles);
  if(data.size() > 0){
    std::vector<Eigen::Triplet<double>> triplets = makeTriplets(data, rows, cols);
    R_lub_cut_wall.setFromTriplets(triplets.begin(), triplets.end());

  } else{
    for(int i = 0; i < 6*num_particles; i++){
      R_lub_cut_wall.insert(i, i) = small;
    }
  }

  C_Lubrication::R_lub = R_lub_cut + R_lub_cut_wall;

  C_Lubrication::Delta_R = R_lub - R_blob;

  std::cout << "C++ R BLOB NNZ: " << R_blob.nonZeros() << std::endl;
  std::cout << "C++ R LUB NNZ: " << R_lub.nonZeros() << std::endl;
  std::cout << "C++ DELTA R NNZ: " << Delta_R.nonZeros() << std::endl;
}

// helper that assembles the COO format Triplets used to make eigen matrices
std::vector<Eigen::Triplet<double>> C_Lubrication::makeTriplets(const std::vector<double>& data, 
                                    const std::vector<int>& rows, const std::vector<int>& cols){

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(data.size());

  for(int i = 0; i < data.size(); i++){
    triplets.push_back(Eigen::Triplet<double>(rows.at(i), cols.at(i), data.at(i)));
  }

  return triplets;
}

Eigen::SparseMatrix<double> C_Lubrication::lub_print(){
  return R_blob;
}

using namespace pybind11::literals;
namespace py = pybind11;
PYBIND11_MODULE(C_Lubrication, m) {
  m.doc() = "C_Lubrication class code";
  py::class_<C_Lubrication>(m, "C_Lubrication")
  .def(py::init<double>()) // C_Lubrication::C_Lubrication constructor
  .def("ResistCOO", &C_Lubrication::ResistCOO)
  .def("ResistCOO_wall", &C_Lubrication::ResistCOO_wall)
  .def("ResistPairSup_py",&C_Lubrication::ResistPairLub_py)
  .def("Set_R_Mats", &C_Lubrication::Set_R_Mats)
  .def("lub_print", &C_Lubrication::lub_print);
}

// int main()
// {
//   C_Lubrication Lub;
//   for (auto row : Lub.mob_scalars_wall_2562) {
//     for (auto el : row) {
//       std::cout << std::setprecision(16) << el << ' ';
//     }
//     std::cout << "\n";
//   }
// //   
// //   for (auto row : Lub.JO_x) {
// //       std::cout << std::setprecision(16) << row << "\n";
// //   }
//   
// //   double r_norm;
// //   double mob_factor[3] = {1.0,1.0,1.0};
// //   r_norm = 2.0084;
// //   //r_norm += 1.0;
// //   //bool inv = false;
// //   Vector3 r_hat(1.0,0.0,0.0);
// //   //Matrix R = Lub.ResistMatrix(r_norm, mob_factor, r_hat, inv, Lub.JO_x, Lub.mob_scalars_JO_11, Lub.mob_scalars_JO_12);
// //   Matrix R = Lub.ATResistMatrix(r_norm, mob_factor, r_hat);
// //   std::cout << std::setprecision(8) << R << std::endl;
//   
// 
// //   while(1){
// //       std::cin >> r_norm;
// //       int i = Lub.FindNearestIndexLower(r_norm, Lub.WS_x);
// //       std::cout << std::setprecision(16) << i << "\n";
// //   }
// //   return 0;
//   
//   double r_norm;
//   while(1){
//       std::cin >> r_norm;
//       double mob_factor[3] = {1.0,1.0,1.0};
//       Matrix R = Lub.WallResistMatrix(r_norm, mob_factor, Lub.Wall_2562_x, Lub.mob_scalars_wall_2562);
//       std::cout << std::setprecision(8) << R << "\n";
//   }
//   return 0;
//   
//   
// }
