CXX ?= g++

CXXFLAGS = -I../ -DPYTHON -DNDEBUG -O3 -ffast-math -march=native -mtune=native -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` mobility.cpp -o mobility_cpp`python3-config --extension-suffix`  -I${CONDA_PREFIX}/include -I${CONDA_PREFIX}/include/eigen3 -fopenmp

ifeq ($(shell uname -s),Darwin)
	CXXFLAGS += -undefined dynamic_lookup
endif

all: mobility.cpp mobility.hpp
	$(CXX) $(CXXFLAGS)
