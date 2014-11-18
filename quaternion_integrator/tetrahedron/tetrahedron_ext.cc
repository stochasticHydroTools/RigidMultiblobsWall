// Functions for tetrahedron simulation written
// in C++ for improved speed.

#include <boost/python.hpp>
#include <math.h>
#include <iostream>

void TestList(boost::python::list list_to_print) {
    // Test using the python list implemented in boost.
    int n  = boost::python::len(list_to_print);
    for (int k = 0; k < n; ++k) {
        std::cout << "x at " << k << " is "
                  << boost::python::extract<double>(list_to_print[k])
                  << std::endl;
    }
}



BOOST_PYTHON_MODULE(tetrahedron_ext)
{
    using namespace boost::python;
    def("test_list", TestList);
}
