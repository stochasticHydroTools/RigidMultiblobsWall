//C version of matlab file for sphere triangulation

// To use compile the code use
// c++ -o create_3d_sphere.exe create_3d_sphere.cpp

// To run the code use
// ./create_3d_sphere.exe center_x center_y center_z DX number_of_layers scaling
// with:
// center_x: the center of the sphere along the x-axis
// Dx: use DX=1
// number_of_layers: for number_of_layers=1 the sphere is an icosahedron. For n_o_l > 1
//                   the sphere is discretize with a higher number of blobs (42, 162, 642...)
// scaling: by default the distance between blobs is = 1 and the sphere radius change
//          use the scaling to change the sphere radius.


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <fstream>
#include <iostream>
using namespace std;

//locate the perpendicular bisector of the great circle joining
//the points whose indices are stored in rows v1 and v2 of x
void 
Locate(
    const int v1, 
    const int v2, 
    const double *x, 
    double *xout)
{
  for (int idir = 0; idir < 3; idir++)
    xout[idir] =(x[v1*3+idir]+x[v2*3+idir])/2;
  double r=0.;

  for (int idir = 0; idir < 3; idir++) r += xout[idir]*xout[idir];

  r=sqrt(r);
  for (int idir = 0; idir < 3; idir++) xout[idir] /=r;
}


//Function to refine triangulation of the sphere
void
RefinePoints(int& nt, int& nv, double *x, int *v)
{
  int vnew[] = {0,0,0};
  const int next[]={1,2,0};
  
  int *a = new int[nv*nv*16];
  memset(a, 0, sizeof(int)*nv*nv);
  int nv0=nv;
  
  for (int tpart=0; tpart<nt; tpart++)
    {
      for (int j=0; j<3; j++)
	{
	  int v1 = v[tpart*3+next[j]];
	  int v2 = v[tpart*3+next[next[j]]];
	  if(a[v1*nv0+v2]==0)
	    {
	      vnew[j]=nv;
	      double xout[3];
	      Locate(v1,v2,x,xout);
	      for (int idir=0; idir<3; idir++) x[nv*3+idir]=xout[idir];
	      a[v1*nv0+v2]=nv;
	      a[v2*nv0+v1]=nv;
	      nv=nv+1;
	    }
	  else
	    vnew[j]=a[v1*nv0+v2];
	}
      v[(1*nt+tpart)*3]   = v[tpart*3];
      v[(1*nt+tpart)*3+1] = vnew[2];
      v[(1*nt+tpart)*3+2] = vnew[1];

      v[(2*nt+tpart)*3]   = v[tpart*3+1];
      v[(2*nt+tpart)*3+1] = vnew[0];
      v[(2*nt+tpart)*3+2] = vnew[2];

      v[(3*nt+tpart)*3]   = v[tpart*3+2];
      v[(3*nt+tpart)*3+1] = vnew[1];
      v[(3*nt+tpart)*3+2] = vnew[0];
      
      for (int idir=0; idir<3; idir++) v[tpart*3+idir] = vnew[idir];
    }
  nt=nt*4;
  delete[] a;
  return;
}


//constructs the vertices of a dodecahedron on the unit sphere
void
Dodecahedron(double *x, int *v)
{
  const double theta = 2.0*M_PI/5.0;
  const double z = cos(theta)/(1.0-cos(theta));
  const double r=sqrt(1-z*z);
    
    x[0] = 0.;
    x[1] = 0.;
    x[2] = 1.;

    x[33]   =  0.;
    x[33+1] =  0.;
    x[33+2] = -1.;

    for (int j=1;j<=5;j++)
      {
	double k=j-.5;
	x[(j)*3]   = r*cos(j*theta);
	x[(j)*3+1] = r*sin(j*theta);
	x[(j)*3+2] = z;
	
	x[(5+j)*3]   = r*cos(k*theta);
	x[(5+j)*3+1] = r*sin(k*theta);
	x[(5+j)*3+2] = -z;
      }
    
    const int input_v[]={1,2,3,
			 1,3,4,
			 1,4,5,
			 1,5,6,
			 1,6,2,
			 2,8,3,
			 3,9,4,
			 4,10,5,
			 5,11,6,
			 6,7,2,
			 8,2,7,
			 9,3,8,
			 10,4,9,
			 11,5,10,
			 7,6,11,
			 7,12,8,
			 8,12,9,
			 9,12,10,
			 10,12,11,
			 11,12,7};

    memcpy(v,input_v,sizeof(input_v));
}


void print_help()
{
  printf("Usage 3D only: Center_coord \t DX \t Num_layers \t min_distance_between_markers \t filled(1/0) \n");
  printf("Default, min_distance=DX and filled=0 it writes only surface layer \n");

  printf("example: ./a.out 1.0 1.0 0.5 1.0 4 1.5 1\n"); 
  exit(1);
}

//create coordinates of point for 3D sphere
int main(int arc, char**argv)
{
  
  if (arc<6) print_help();
  double center[3];
  center[0]=strtod(argv[1],NULL);
  center[1]=strtod(argv[2],NULL);
  center[2]=strtod(argv[3],NULL);

  double Dx = strtod(argv[4],NULL);
  int num_layers = strtol(argv[5],NULL,10); // Number of layers in sphere

  double DXLayer;
  if (arc>6) DXLayer = strtod(argv[6],NULL);
  else DXLayer=Dx;

  int filled;
  if (arc>7) filled = strtod(argv[7],NULL);
  else filled=0;

  //since pow works only for double type
  int factor = 4;
  for (int ipow=0;ipow<num_layers;ipow++) factor *= 4;
  const int nvmax = 12*factor;
  const int ntmax=20*factor;

  //initilize nt, nv
  int nv = 12;
  int nt = 20;
  
  //total number of vertex in file 
  int total_vertex=0;
  //number of vertex for each layer
  int num_vertex[num_layers];
  num_vertex[0]=12; //starting  with dodecahedron

  //allocate space for x and v
  double *x = new double[nvmax*3];

  int *v    = new int[ntmax*3];

  Dodecahedron(x,v);
 
  ofstream vertex;
  vertex.open ("shell_3d.vertex");

  for(int ipart=0;ipart<nt*3;ipart++) v[ipart] -=1;

  for (int ilayer=1;ilayer<num_layers;ilayer++)
    {

      RefinePoints(nt, nv, x, v);

      num_vertex[ilayer]=nv;

    }


  //checks if we need only surface layer
  if  (filled) 
      for (int ilayer=0;ilayer<num_layers;ilayer++) total_vertex +=num_vertex[ilayer]; 
  else
      total_vertex = num_vertex[num_layers-1];
  


  if  (filled) 
  {
      vertex<<total_vertex+1<< " " << Dx << endl;
      //storing the center of sphere.
      for(int idir=0;idir<3;idir++) vertex << center[idir]<< "\t";
      vertex<<endl;
      total_vertex++;
  }
  else
  {
      vertex<<total_vertex<< "  " << Dx << endl;
  }
  vertex<<scientific;
  vertex.precision(15);

  double R_scale[num_layers];

  for (int ilayer=0;ilayer<num_layers;ilayer++)
  {
      double min_dist=total_vertex*Dx;
      //check if we need to save only surface layer
      if  (!filled) 
      {
	  ilayer = num_layers-1;
      }
      for(int ipart=0;ipart<num_vertex[ilayer];ipart++) 
      {
	  for(int jpart=ipart+1;jpart<num_vertex[ilayer];jpart++) 
	  {
	      double dist=0.;
	      for(int idir=0;idir<3;idir++) 
		  dist +=(x[ipart*3+idir]-x[jpart*3+idir])*(x[ipart*3+idir]-x[jpart*3+idir]);
	      dist=sqrt(dist);
	      if (min_dist>dist) min_dist=dist;
	  }
      }
//check if a radius less than required distance
      R_scale[ilayer] = max((DXLayer/min_dist), (ilayer>0)?(DXLayer+R_scale[ilayer-1]):DXLayer);
      cout.precision(12);
      cout<<ilayer<<"-layer min_dist on the unit sphere="<<min_dist<<"   R_scale["<<ilayer<<"]="<<R_scale[ilayer]<<endl;
  }
  cout<<"Total number of vertices is "<<total_vertex<<endl;
  

  double max_distance = 0.0; // Estimate radius of sphere (required box size)
  
  for (int ilayer=0;ilayer<num_layers;ilayer++)
  {
      //check if we need to save only surface layer
      if  (!filled) 
      {
	  ilayer = num_layers-1;
      }
      
      for(int ipart=0;ipart<num_vertex[ilayer];ipart++) 
      {
          double displ[3], distance;
          distance=0;
          for(int idir=0;idir<3;idir++) {
            displ[idir]=(R_scale[ilayer])*x[ipart*3+idir];
            distance = distance + displ[idir]*displ[idir];
          }
          distance=sqrt(distance);
          if(distance>max_distance) max_distance=distance;
	  for(int idir=0;idir<3;idir++) vertex<<center[idir]+displ[idir]<<"\t";
	  vertex<<endl;
	}
  }
  vertex.close();
  printf("Largest distance of marker to center ~ radius of sphere = %lf \n", max_distance);
  delete[] x;
  delete[] v;
} 
