# location of the Python header files
PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# location of the Boost Python include files and library
BOOST_INC = /home/bsprinkle/SOFTWARES/LIBRARIESIBAMR/Boost/1.57.0/include
BOOST_LIB = /home/bsprinkle/SOFTWARES/LIBRARIESIBAMR/Boost/1.57.0/libs

# Location of Fortran libraries,
# uncomment to use the Fast Multipole Metho library dstfmmlib3d-1.2
# FORTRAN_LIB = --link-fopenmp -L/home/bsprinkle/SOFTWARES/LIBRARIESIBAMR/FMM3D/1.2/stfmmlib3d-1.2/ -lstfmm3d 

# 
TARGET = mobility_ext
TARGET_FORTRAN = mobility_fmm

all: $(TARGET).so
ifneq ($(FORTRAN_LIB),)
all: $(TARGET_FORTRAN).so
endif

$(TARGET_FORTRAN).so: $(TARGET_FORTRAN).f90
	f2py $(FORTRAN_LIB) -c $(TARGET_FORTRAN).f90 -m $(TARGET_FORTRAN)

$(TARGET).so: $(TARGET).o
	g++ -shared  -dynamiclib $(TARGET).o -L$(BOOST_LIB)  -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $(TARGET).so

$(TARGET).o: $(TARGET).cc
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c $(TARGET).cc

clean: 
	rm -f $(TARGET).o $(TARGET).so $(TARGET_FORTRAN).o $(TARGET_FORTRAN).so
