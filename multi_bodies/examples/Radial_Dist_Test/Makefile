CXX=g++
CXXFLAGS= -std=c++11 -fopenmp -O3
SOURCES=gr_pseudo2D_single_blob.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=gr_pseudo2D

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-rm $(OBJECTS) $(EXECUTABLE)
