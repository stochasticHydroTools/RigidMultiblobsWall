all: c_fibers_obj c_rigid_obj

c_fibers_obj: c_fibers_obj.cpp
	c++ -O3 -shared c_fibers_obj.cpp ../libMobility/solvers/PSE/mobility.so ../libMobility/solvers/NBody/mobility.so -o c_fibers_obj.so -std=c++14 -fPIC -I.. -I../libMobility/include -I/usr/include/eigen3 -DDOUBLE_PRECISION  -llapack -lblas -llapacke `python3 -m pybind11 --includes` 
c_rigid_obj: c_rigid_obj.cpp
	c++ -O3 -shared c_rigid_obj.cpp ../libMobility/solvers/PSE/mobility.so ../libMobility/solvers/NBody/mobility.so ../libMobility/solvers/DPStokes/mobility.so -o c_rigid_obj.so -std=c++14 -fPIC -I.. -I../libMobility/include -I/usr/include/eigen3 -DDOUBLE_PRECISION -llapack -lblas -llapacke `python3 -m pybind11 --includes`
clean:
	rm *.so

