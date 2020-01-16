#include <cmath>
#include <cstdio>
#include <iostream>
#ifdef PYTHON
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#endif

#include "forces.hpp"

dvec blob_blob_force(Eigen::Ref<dvecvec> r_vectors_in, Eigen::Ref<dvec> L,
                     double eps, double b, double a) {
    const int N = r_vectors_in.size() / 3;
    Eigen::Map<dvecvec> r_vectors(r_vectors_in.data(), N, 3);
    dvec force = dvec(N * 3).setZero();

#pragma omp parallel for reduction(+ : force) schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            Eigen::Vector3d dr = r_vectors.row(j) - r_vectors.row(i);
            for (int k = 0; k < 3; ++k){
                if (L[k] > 0)
                    dr[k] -= int(dr[k] / L[k] +
                                 0.5 * (int(dr[k] > 0) - int(dr[k] < 0))) *
                             L[k];
            }
            double r_norm = dr.norm();
            double f0 =
                r_norm > 2 * a
                    ? -((eps / b) * exp(-(r_norm - 2.0 * a) / b) / r_norm)
                    : -((eps / b) / std::max(r_norm, 1e-25));

            for (int k = 0; k < 3; ++k)
                force[i * 3 + k] += f0 * dr[k];
            for (int k = 0; k < 3; ++k)
                force[j * 3 + k] -= f0 * dr[k];
        }
    }

    return force;
}

#ifdef PYTHON
PYBIND11_MODULE(forces_cpp, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("blob_blob_force",
          &blob_blob_force, "Calculate blob-blob interaction");
}
#endif
