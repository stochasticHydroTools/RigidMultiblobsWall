// Compute the radial distribution function in a system
// // periodic along the x and y axes and dimensions [lx,ly].
// // Compute 3D distance between bodies but normalize density
// // as if it was a 2D system. This is appropriate for quasi-2D
// // systems, e.g., particles sedimented over a wall. How to use:
// //
// // radialDistribution.exe fileInput np lx ly lz > fileOutput
// //
// //

#include <iostream>
#include <string>
#include <stdlib.h> //for pseudorandom numbers
#include <time.h>
#include <fstream>
#include <math.h>
using namespace std;


int main(  int argc, char* argv[]){
  string word;
  ifstream fileinput(argv[1]);
  int np = atoi(argv[2]);
  double lx = atof(argv[3]);
  double ly = atof(argv[4]);
  double lz = atof(argv[5]);
  int numberOfBins = 2000; //1000

  double r,dr,t,xij,yij,zij, aux;
  dr = lx / (2.0 * numberOfBins);
  double x[np], y[np], z[np];
  int bin;
  int n=0;
  int hist[numberOfBins];
  for(int i=0;i<numberOfBins;i++)
    hist[i]=0;

  int skip=0;
  int step = 0;
  while(!fileinput.eof()){
    fileinput >> t;
    n++;
    for(int i=0;i<np;i++){
      fileinput >> x[i] >> y[i] >> z[i] >> aux >> aux >> aux >> aux;
    }
    if(n>skip){
      for(int i=0;i<np-1;i++){
    for(int j=i+1;j<np;j++){
      xij = x[i] - x[j];
      xij = xij - (int(xij/lx + 0.5*((xij>0)-(xij<0)))) * lx;
      yij = y[i] - y[j];
      yij = yij - (int(yij/ly + 0.5*((yij>0)-(yij<0)))) * ly;
      zij = z[i] - z[j];
      //zij = zij - (int(zij/lz + 0.5*((zij>0)-(zij<0)))) * lz;
     
      r = sqrt(xij*xij + yij*yij + zij*zij);
      bin = int(r/dr);
      if(bin<numberOfBins){
        hist[bin] = hist[bin] + 2;
      }
    }
      }
    }
    step++;
    //cout << step << endl;
  }


 
  double rLow, rUp, nIdeal, dens, pi, constant;
  pi = 4 * atan(1);
  dens = np / (lx*ly);
  constant = pi * dens;
  for(int i=0;i<numberOfBins;i++){
    rLow = i * dr;
    rUp = rLow + dr;
    nIdeal = constant * (pow(rUp,2) - pow(rLow,2));
    cout << (i+0.5)*dr << " " << hist[i]/((n-skip)*np*nIdeal) << " " << hist[i] << endl;
  }

 
}

