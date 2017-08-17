// Functions to write VTK files from C++ 
#include <boost/python.hpp>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>
#include "visit_writer.h"
#include "visit_writer.c"

namespace bp = boost::python;

void visitWriterInterface(std::string name,
			  int format,
                          bp::numeric::array dims,
			  bp::numeric::array x,
			  bp::numeric::array y,
			  bp::numeric::array z,
			  int nvars,
			  bp::numeric::array vardims,
			  bp::numeric::array centering,
			  bp::list varnames,
			  bp::list variables){

  // Copy python variables to C++ variables
  int dims_array[3];
  for(int i=0; i<3; i++)
    dims_array[i] = bp::extract<int>(dims[i]);
  int vardims_array[1];
  vardims_array[0] = bp::extract<int>(vardims[0]);
  int centering_array[1];
  centering_array[0] = bp::extract<int>(centering[0]);
  std::string varnames_array[1];
  varnames_array[0] = bp::extract<std::string>(varnames[0]);
  int sizeNameVelocity = varnames_array[0].size();
  char nameVelocity[sizeNameVelocity]; 
  varnames_array[0].copy(nameVelocity, sizeNameVelocity);
  char **varnames_char = new char* [1];
  varnames_char[0] = &nameVelocity[0];
  bp::numeric::array variables_velocity = bp::extract<bp::numeric::array>(variables[0]);
  int nCells = (dims_array[0]-1)*(dims_array[1]-1)*(dims_array[2]-1);
  double *velocity = new double [3*nCells];
  for(int i=0; i<nCells*3;i++){
    velocity[i] = bp::extract<double>(variables_velocity[i]);
  }
  double **vars;
  vars = new double* [1];
  vars[0] = velocity;
  
  int nx =dims_array[0];
  double *xmesh = new double [nx];
  for(int i=0; i<nx;i++){
    xmesh[i] = bp::extract<double>(x[i]);
  }
  int ny =dims_array[1];
  double *ymesh = new double [ny];
  for(int i=0; i<ny;i++){
    ymesh[i] = bp::extract<double>(y[i]);
  }
  int nz =dims_array[2];
  double *zmesh = new double [nz];
  for(int i=0; i<nz;i++){
    zmesh[i] = bp::extract<double>(z[i]);
  }
  

  

  // Print variables
  if(0){
      std::cout << std::endl << "visitWriterInterface " << std::endl;
      std::cout << "name: " << name << std::endl;
      std::cout << "format: " << format << std::endl;
      std::cout << "dims: " << dims_array[0] << "  " << dims_array[1] << "  " << dims_array[2] << std::endl;
      std::cout << "nvars: " << nvars << std::endl;
      std::cout << "vardims: " << vardims_array[0] << std::endl;
      std::cout << "centering: " << centering_array[0] << std::endl;
      std::cout << "varnames: " << varnames_array[0] << std::endl;
      if(0)
	for(int i=0; i<nCells*3;i++){
	  std::cout << "velocity " << i << "  " << velocity[i] << std::endl;
	}
      std::cout << std::endl;
  }

  // Call visit_writer
  /*Use visit_writer to write a regular mesh with data. */
  write_rectilinear_mesh(name.c_str(),    // Output file
                         format,          // 0=ASCII,  1=Binary
                         dims_array,      // {mx, my, mz}
                         xmesh,           
                         ymesh,
                         zmesh,
                         nvars,           // number of variables
                         vardims_array,   // Size of each variable, 1=scalar, velocity=3*scalars
                         centering_array, // 
                         varnames_char,   //
                         vars);

  // Free memory in the heap
  delete[] velocity;
  delete[] vars;
  delete[] varnames_char;
}


BOOST_PYTHON_MODULE(visit_writer_interface)
{
  using namespace boost::python;
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  def("visit_writer_interface", visitWriterInterface);
}

