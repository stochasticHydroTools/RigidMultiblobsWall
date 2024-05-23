cmake_minimum_required(VERSION 3.0)
project(your_project_name)

set(CMAKE_CXX_STANDARD 14)

set(SOURCES
    c_fibers_obj.cpp
)

set(CONDA_PREFIX_PATH "$ENV{CONDA_PREFIX}")
# find_library(MOBILITY_LIBRARY_PSE NAMES mobility PATHS ${CONDA_PREFIX_PATH}/libmobility/solvers/PSE)
# find_library(MOBILITY_LIBRARY_NBody NAMES mobility PATHS ${CONDA_PREFIX_PATH}/libmobility/solvers/NBody)
find_library(libMobility_PSE libMobility_NBody)

# find_package(LAPACK REQUIRED)
# find_package(LAPACKE REQUIRED)
# find_package(BLAS REQUIRED)

find_package(BLAS)
if(BLAS_FOUND)
  message("mkl environment detected")
  add_compile_definitions(PUBLIC USE_MKL)
  link_libraries(
    BLAS::BLAS
  )
  find_path(BLAS_INCLUDE_DIRS mkl.h
    $ENV{CONDA_PREFIX}/include
    /usr/include
    /usr/local/include
    $ENV{MKLROOT}/include
    $ENV{BLAS_HOME}/include
  )
else()
  unset(BLA_VENDOR)
  find_package(LAPACK REQUIRED)
  find_package(LAPACKE REQUIRED)
  find_package(BLAS REQUIRED)
  link_libraries(${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})
  find_path(BLAS_INCLUDE_DIRS cblas.h
    $ENV{CONDA_PREFIX}/include
    /usr/include
    /usr/local/include
    $ENV{MKLROOT}/include
    $ENV{BLAS_HOME}/include
  )
  find_path(LAPACKE_INCLUDE_DIRS lapacke.h
    $ENV{CONDA_PREFIX}/include
    /usr/include
    /usr/local/include
    $ENV{MKLROOT}/include
    $ENV{LAPACKE_HOME}/include
  )

endif()

find_package (Eigen3 REQUIRED NO_MODULE)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

include_directories(${BLAS_INCLUDE_DIRS} ${LAPACKE_INCLUDE_DIRS} ${CONDA_PREFIX_PATH}/libMobility/include)



if(NOT Python3_SOABI)
  message(FATAL_ERROR "Python3_SOABI is empty. Set it to the output of 'python3-config --extension-suffix'")
endif()

# Add executable for c_fibers_obj
add_library(c_fibers_obj.${Python3_SOABI} SHARED c_fibers_obj.cpp)
set_target_properties(c_fibers_obj.${Python3_SOABI} PROPERTIES PREFIX "")
# add_library(c_fibers_obj MODULE ${SOURCES})
target_compile_options(c_fibers_obj.${Python3_SOABI} PRIVATE -fPIC ${DOUBLEPRECISION})
target_link_libraries(c_fibers_obj.${Python3_SOABI} PRIVATE libMobility_PSE libMobility_NBody pybind11::pybind11 Eigen3::Eigen)

install(TARGETS c_fibers_obj.${Python3_SOABI} DESTINATION ${Python3_SITEARCH}/fibers)

# Set the output name for the shared library
#set_target_properties(c_fibers_obj PROPERTIES OUTPUT_NAME c_fibers_obj)
