CXX ?= g++

CXXFLAGS = -I../ -DPYTHON -DNDEBUG -O3 -ffast-math -march=native -mtune=native -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` forces.cpp -o forces_cpp`python3-config --extension-suffix`  -I${CONDA_PREFIX}/include -I${CONDA_PREFIX}/include/eigen3 -fopenmp

ifeq ($(shell uname -s),Darwin)
	CXXFLAGS += -undefined dynamic_lookup
endif

all: forces.cpp forces.hpp
	$(CXX) $(CXXFLAGS)
