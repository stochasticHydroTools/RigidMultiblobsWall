
CXXFLAGS +=  -I${CONDA_PREFIX}/include/eigen3 -fopenmp -I../ -O3 -ffast-math -march=native -mtune=native -Wall -std=c++17
LINKFLAGS += -O3 -fopenmp

main: main.o mobility.o forces.o
	g++ $(LINKFLAGS) main.o forces.o mobility.o -fopenmp -o main

main.o: main.cpp
	g++ $(CXXFLAGS) -c main.cpp -o main.o

forces.o: ../multi_bodies/forces.hpp ../multi_bodies/forces.cpp
	g++ $(CXXFLAGS) -c ../multi_bodies/forces.cpp -o forces.o

mobility.o: ../mobility/mobility.hpp ../mobility/mobility.cpp
	g++ $(CXXFLAGS) -c ../mobility/mobility.cpp -o mobility.o

clean:
	rm -f *.o main
