#ifndef MOBILITY_HPP
#define MOBILITY_HPP

#include <Eigen/Core>
#include <omp.h>

// Some convenience types
typedef Eigen::ArrayXd dvec;
typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    dvecvec;

#pragma omp declare reduction (+: dvec: omp_out=omp_out+omp_in)\
     initializer(omp_priv=dvec::Zero(omp_orig.size()))

dvec blob_blob_force(Eigen::Ref<dvecvec> r_vectors, Eigen::Ref<dvec> L,
                     double eps, double b, double a);

#endif
