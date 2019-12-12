
all: mobility.cpp mobility.hpp
	g++ -DPYTHON -O3 -ffast-math -march=native -mtune=native -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` mobility.cpp -o mobility_cpp`python3-config --extension-suffix` -I${CONDA_PREFIX}/include/eigen3 -fopenmp
