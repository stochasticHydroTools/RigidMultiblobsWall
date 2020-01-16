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

dvecvec rotne_prager_tensor(Eigen::Ref<dvecvec> r_vectors_in, double eta,
                            double a);

dvec single_wall_mobility_trans_times_force(Eigen::Ref<dvecvec> r_vectors,
                                            Eigen::Ref<dvecvec> force, double eta,
                                            double a, Eigen::Ref<dvec> L);

#endif
