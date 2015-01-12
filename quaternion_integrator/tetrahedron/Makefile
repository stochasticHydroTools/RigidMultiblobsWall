# location of the Python header files
 
PYTHON_VERSION = 2.6
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
 
# location of the Boost Python include files and library
 
BOOST_INC = /usr/include  # Original version
#BOOST_INC = /usr/local/Cellar/boost/1.56.0/include # Mac version.
BOOST_LIB = /usr/lib  #Original version
#BOOST_LIB = /usr/local/Cellar/boost-python/1.56.0/lib # Mac version.
 
# compile mesh classes
TARGET = tetrahedron_ext
 
$(TARGET).so: $(TARGET).o
	g++ -shared  -dynamiclib $(TARGET).o -L$(BOOST_LIB)  -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $(TARGET).so
 
$(TARGET).o: $(TARGET).cc
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c $(TARGET).cc

clean: 
	rm -f $(TARGET).o $(TARGET).so
