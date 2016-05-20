# location of the Python header files
PYTHON_VERSION = 2.6
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# location of the Boost Python include files and library
BOOST_INC = /usr/include/boost
BOOST_LIB = /usr/lib64/boost/


# compile mesh classes
TARGET = visit_writer_interface


$(TARGET).so: $(TARGET).o
	g++ -shared  -dynamiclib $(TARGET).o -L$(BOOST_LIB)  -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $(TARGET).so

$(TARGET).o: $(TARGET).cc
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c $(TARGET).cc
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c visit_writer.c

clean: 
	rm -f $(TARGET).o $(TARGET).so
