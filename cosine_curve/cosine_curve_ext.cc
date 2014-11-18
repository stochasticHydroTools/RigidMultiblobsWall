// Surface constraint for constrainted integrator.
// Written in C++ to see if it's faster.

#include <boost/python.hpp>
#include <math.h>

#define PI 3.141592653

double CosineConstraint(double x, double y) {
    double r_squared = x*x + y*y;
    double theta = atan(y/x);
    if (x < 0) theta += PI;
    return r_squared - pow(0.25*cos(3.0*theta) + 1.0, 2);
}

BOOST_PYTHON_MODULE(cosine_curve_ext)
{
    using namespace boost::python;
    def("cosine_constraint", CosineConstraint);
}
