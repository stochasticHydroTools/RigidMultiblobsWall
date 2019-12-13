#include <cmath>
#include <cstdio>
#include <iostream>
#ifdef PYTHON
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#endif

#include <mobility/mobility.hpp>

dvecvec shift_heights(Eigen::Ref<dvecvec> r_vectors, double blob_radius) {
    dvecvec r_shifted = r_vectors;
    for (int i = 0; i < r_vectors.rows(); ++i) {
        if (r_shifted(i, 2) < blob_radius)
            r_shifted(i, 2) = blob_radius;
    }
    return r_shifted;
}

std::tuple<dvecvec, bool> damping_matrix_B(Eigen::Ref<dvecvec> r_vectors,
                                           double blob_radius) {
    // Return sparse diagonal matrix with components
    // B_ii = 1.0               if z_i >= blob_radius
    // B_ii = z_i / blob_radius if z_i < blob_radius

    // It is used to compute positive definite mobilities
    // close to the wall.

    dvec B(r_vectors.size());
    B.fill(1);

    bool overlap = false;
    for (int k = 0; k < r_vectors.rows(); ++k) {
        if (r_vectors(k, 2) < blob_radius) {
            B[k * 3] = r_vectors(k, 2) / blob_radius;
            B[k * 3 + 1] = r_vectors(k, 2) / blob_radius;
            B[k * 3 + 2] = r_vectors(k, 2) / blob_radius;
            overlap = true;
        }
    }

    // FIXME: should be sparse matrix
    return std::make_tuple(B, overlap);
    // return (scipy.sparse.dia_matrix((B, 0), shape=(B.size, B.size)),
    // overlap);
}

dvecvec single_wall_fluid_mobility(Eigen::Ref<dvecvec> r_vectors_in, double eta,
                                   double a) {
    // Mobility for particles near a wall.  This uses the expression from
    // the Swan and Brady paper for a finite size particle, as opposed to the
    // Blake paper point particle result.

    // For blobs overlaping the wall we use
    // Compute M = B^T * M_tilde(z_effective) * B.
    // Get effective height;

    Eigen::Map<dvecvec> r_vectors(r_vectors_in.data(), r_vectors_in.size() / 3,
                                  3);

    dvecvec r_vectors_effective = shift_heights(r_vectors, a);
    // Compute damping matrix B;
    auto [B_damp, overlap] = damping_matrix_B(r_vectors, a);
    // We add the corrections from the appendix of the paper to the unbounded
    // mobility.
    dvecvec fluid_mobility = rotne_prager_tensor(r_vectors_effective, eta, a);

    // Extract variables;
    int N = r_vectors.size() / 3;
    dvec x = r_vectors_effective.col(0);
    dvec y = r_vectors_effective.col(1);
    dvec z = r_vectors_effective.col(2);

    // Compute distances between blobs;
    dvecvec dx(N, N);
    dvecvec dy(N, N);
    dvecvec dz(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            dx(i, j) = (x[i] - x[j]) / a;
            dy(i, j) = (y[i] - y[j]) / a;
            dz(i, j) = (z[i] + z[j]) / a;
        }
    }
    dvecvec dr = (dx.pow(2) + dy.pow(2) + dz.pow(2)).sqrt();

    dvecvec h_hat = z.replicate(1, N) / (a * dz);
    dvec h = z / a;
    dvecvec ex = dx / dr;
    dvecvec ey = dy / dr;
    dvecvec ez = dz / dr;

    // Compute scalar functions, the mobility is;
    // M = A*delta_ij + B*e_i*e_j + C*e_i*delta_3j + D*delta_i3*e_j +
    // E*delta_i3*delta_3j ;
    double factor = 1.0 / (6.0 * M_PI * eta * a);

    // Allocate memory;
    dvecvec M = dvecvec(N * 3, N * 3).setZero();
    dvecvec A = dvecvec(N, N).setZero();
    dvecvec B = dvecvec(N, N).setZero();
    dvecvec C = dvecvec(N, N).setZero();
    dvecvec D = dvecvec(N, N).setZero();
    dvecvec E = dvecvec(N, N).setZero();

    // Self-mobility terms;
    dvec A_vec = -0.0625 * (9.0 / h - 2.0 / h.pow(3) + 1.0 / h.pow(5));
    dvec E_vec = -A_vec - 0.125 * (9.0 / h - 4.0 / h.pow(3) + 1.0 / h.pow(5));

    for (int i = 0; i < N; ++i) {
        M(i * 3, i * 3) += A_vec(i);
        M(i * 3 + 1, i * 3 + 1) += A_vec(i);
        M(i * 3 + 2, i * 3 + 2) += E_vec(i) + A_vec(i);
    }

    // Particle-particle terms;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (dr(i, j) > 1E-12 * a && i != j) {
                double drij3 = dr(i, j) * dr(i, j) * dr(i, j);
                double drij5 = dr(i, j) * dr(i, j) * drij3;
                double ez2 = ez(i, j) * ez(i, j);
                A(i, j) =
                    -0.25 *
                    (3.0 * (1.0 + 2 * h_hat(i, j) * (1.0 - h_hat(i, j)) * ez2) /
                         dr(i, j) +
                     2.0 * (1 - 3.0 * ez2) / drij3 -
                     2.0 * (1 - 5.0 * ez2) / drij5);

                B(i, j) =
                    -0.25 *
                    (3.0 *
                         (1.0 - 6.0 * h_hat(i, j) * (1.0 - h_hat(i, j)) * ez2) /
                         dr(i, j) -
                     6.0 * (1.0 - 5.0 * ez2) / drij3 +
                     10.0 * (1.0 - 7.0 * ez2) / drij5);

                C(i, j) =
                    0.5 * ez(i, j) *
                    (3.0 * h_hat(i, j) *
                         (1.0 - 6.0 * (1.0 - h_hat(i, j)) * ez2) / dr(i, j) -
                     6.0 * (1.0 - 5.0 * ez2) / drij3 +
                     10.0 * (2.0 - 7.0 * ez2) / drij5);

                D(i, j) = 0.5 * ez(i, j) *
                          (3.0 * h_hat(i, j) / dr(i, j) - 10.0 / drij5);

                E(i, j) = -(3.0 * h_hat(i, j) * h_hat(i, j) * ez2 / dr(i, j) +
                            3.0 * ez2 / drij3 + (2.0 - 15.0 * ez2) / drij5);
            }
        }
    }

    // Build mobility matrix of size 3N \times 3N;
    dvecvec t1 = A + B * ex * ex;
    dvecvec t2 = B * ex * ey;
    dvecvec t3 = B * ex * ez + C.transpose() * ex;

    dvecvec t4 = B * ey * ex;
    dvecvec t5 = A + B * ey * ey;
    dvecvec t6 = B * ey * ez + C.transpose() * ey;

    dvecvec t7 = B * ez * ex + D.transpose() * ex;
    dvecvec t8 = B * ez * ey + D.transpose() * ey;
    dvecvec t9 = A + B * ez * ez + C.transpose() * ez + D.transpose() * ez +
                 E.transpose();

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            M(i * 3 + 0, j * 3 + 0) += t1(i, j);
            M(i * 3 + 0, j * 3 + 1) += t2(i, j);
            M(i * 3 + 0, j * 3 + 2) += t3(i, j);

            M(i * 3 + 1, j * 3 + 0) += t4(i, j);
            M(i * 3 + 1, j * 3 + 1) += t5(i, j);
            M(i * 3 + 1, j * 3 + 2) += t6(i, j);

            M(i * 3 + 2, j * 3 + 0) += t7(i, j);
            M(i * 3 + 2, j * 3 + 1) += t8(i, j);
            M(i * 3 + 2, j * 3 + 2) += t9(i, j);
        }
    }

    M *= factor;
    M += fluid_mobility;

    // FIXME: Use damping results
    // Compute M = B^T * M_tilde * B;
    // if (overlap)
    //     return B_damp.dot((B_damp.dot(M.T)).T);
    // else
    return M;
}

dvecvec rotne_prager_tensor(Eigen::Ref<dvecvec> r_vectors_in, double eta,
                            double a) {
    // Extract variables
    Eigen::Map<dvecvec> r_vectors(r_vectors_in.data(), r_vectors_in.size() / 3,
                                  3);

    dvec x = r_vectors.col(0);
    dvec y = r_vectors.col(1);
    dvec z = r_vectors.col(2);

    // Compute distances between blobs
    dvecvec dx(x.size(), x.size());
    dvecvec dy(y.size(), y.size());
    dvecvec dz(z.size(), z.size());
    for (int i = 0; i < x.size(); ++i) {
        for (int j = 0; j < x.size(); ++j) {
            dx(i, j) = x[j] - x[i];
            dy(i, j) = y[j] - y[i];
            dz(i, j) = z[j] - z[i];
        }
    }
    dvecvec dr = (dx.pow(2) + dy.pow(2) + dz.pow(2)).sqrt();

    // Compute scalar functions f(r) and g(r)
    double factor = 1.0 / (6.0 * M_PI * eta);
    dvecvec fr = dvecvec(x.size(), x.size()).setZero();
    dvecvec gr = dvecvec(x.size(), x.size()).setZero();

    for (int i = 0; i < x.size(); ++i) {
        for (int j = 0; j < x.size(); ++j) {
            if (dr(i, j) > 2.0 * a) {
                double drij3 = dr(i, j) * dr(i, j) * dr(i, j);
                double drij5 = dr(i, j) * dr(i, j) * drij3;
                fr(i, j) = factor * (0.75 / dr(i, j) + a * a / (2.0 * drij3));
                gr(i, j) = factor * (0.75 / drij3 - 1.5 * a * a / drij5);
            } else if (dr(i, j) == 0.0) {
                fr(i, j) = factor / a;
            } else {
                fr(i, j) = factor * (1.0 / a - 0.28125 * dr(i, j) / (a * a));
                gr(i, j) = factor * (3.0 / (32.0 * a * a * dr(i, j)));
            }
        }
    }

    // Build mobility matrix of size 3N \times 3N
    dvecvec M = dvecvec(r_vectors.size(), r_vectors.size());

    dvecvec t1 = fr + gr * dx * dx;
    dvecvec t2 = gr * dx * dy;
    dvecvec t3 = gr * dx * dz;

    dvecvec t4 = gr * dy * dx;
    dvecvec t5 = fr + gr * dy * dy;
    dvecvec t6 = gr * dy * dz;

    dvecvec t7 = gr * dz * dx;
    dvecvec t8 = gr * dz * dy;
    dvecvec t9 = fr + gr * dz * dz;
    for (int i = 0; i < x.size(); ++i) {
        for (int j = 0; j < x.size(); ++j) {
            M(i * 3, j * 3) = t1(i, j);
            M(i * 3, j * 3 + 1) = t2(i, j);
            M(i * 3, j * 3 + 2) = t3(i, j);

            M(i * 3 + 1, j * 3) = t4(i, j);
            M(i * 3 + 1, j * 3 + 1) = t5(i, j);
            M(i * 3 + 1, j * 3 + 2) = t6(i, j);

            M(i * 3 + 2, j * 3) = t7(i, j);
            M(i * 3 + 2, j * 3 + 1) = t8(i, j);
            M(i * 3 + 2, j * 3 + 2) = t9(i, j);
        }
    }

    return M;
}

dvec single_wall_mobility_trans_times_force(Eigen::Ref<dvecvec> r_vectors_in,
                                            Eigen::Ref<dvecvec> force_in,
                                            double eta, double a,
                                            Eigen::Ref<dvec> L) {
    const int N = r_vectors_in.size() / 3;
    Eigen::Map<dvecvec> r_vectors(r_vectors_in.data(), N, 3);
    Eigen::Map<dvecvec> force(force_in.data(), N, 3);

    const double fourOverThree = 4.0 / 3.0;
    const double inva = 1.0 / a;
    const double norm_fact_f = 1.0 / (8.0 * M_PI * eta * a);

    dvec u = dvec(N * 3).setZero();

    // Loop over image boxes and then over particles
    // TODO: Add PBC!
#pragma omp parallel for reduction(+ : u) schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        { // self interaction
            const double invZi = a / r_vectors(i, 2);
            const double invZi3 = invZi * invZi * invZi;
            const double invZi5 = invZi3 * invZi * invZi;

            double Mxx =
                fourOverThree - (9.0 * invZi - 2.0 * invZi3 + invZi5) / 12.0;
            double Myy =
                fourOverThree - (9.0 * invZi - 2.0 * invZi3 + invZi5) / 12.0;
            double Mzz =
                fourOverThree - (9.0 * invZi - 4.0 * invZi3 + invZi5) / 6.0;

            u[i * 3 + 0] += Mxx * force(i, 0) * norm_fact_f;
            u[i * 3 + 1] += Myy * force(i, 1) * norm_fact_f;
            u[i * 3 + 2] += Mzz * force(i, 2) * norm_fact_f;
        }

        for (int j = i + 1; j < N; ++j) {
            double Mxx, Mxy, Mxz;
            double Myx, Myy, Myz;
            double Mzx, Mzy, Mzz;

            // Compute scaled vector between particles i and j
            Eigen::Vector3d dr = inva * (r_vectors.row(i) - r_vectors.row(j));

            // Normalize distance with hydrodynamic radius
            const double r = dr.norm();
            const double r2 = r * r;

            // TODO: We should not divide by zero
            const double invr = 1.0 / r;
            const double invr2 = invr * invr;

            if (r > 2) {
                const double c1 = 1.0 + 2.0 / (3.0 * r2);
                const double c2 = (1.0 - 2.0 * invr2) * invr2;
                Mxx = (c1 + c2 * dr[0] * dr[0]) * invr;
                Mxy = (c2 * dr[0] * dr[1]) * invr;
                Mxz = (c2 * dr[0] * dr[2]) * invr;
                Myy = (c1 + c2 * dr[1] * dr[1]) * invr;
                Myz = (c2 * dr[1] * dr[2]) * invr;
                Mzz = (c1 + c2 * dr[2] * dr[2]) * invr;
            } else {
                const double c1 =
                    fourOverThree * (1.0 - 0.28125 * r); //  9/32 = 0.28125
                const double c2 =
                    fourOverThree * 0.09375 * invr; // 3/32 = 0.09375
                Mxx = c1 + c2 * dr[0] * dr[0];
                Mxy = c2 * dr[0] * dr[1];
                Mxz = c2 * dr[0] * dr[2];
                Myy = c1 + c2 * dr[1] * dr[1];
                Myz = c2 * dr[1] * dr[2];
                Mzz = c1 + c2 * dr[2] * dr[2];
            }
            Myx = Mxy;
            Mzx = Mxz;
            Mzy = Myz;

            // Wall correction
            dr[2] = (r_vectors(i, 2) + r_vectors(j, 2)) / a;
            double hj = r_vectors(j, 2) / a;

            const double h_hat = hj / dr[2];
            const double invR = 1.0 / dr.norm();
            const double ex = dr[0] * invR;
            const double ey = dr[1] * invR;
            const double ez = dr[2] * invR;
            const double ez2 = ez * ez;
            const double invR3 = invR * invR * invR;
            const double invR5 = invR3 * invR * invR;

            const double t1 = (1.0 - h_hat) * ez2;
            const double fact1 = -(3.0 * (1.0 + 2.0 * h_hat * t1) * invR +
                                   2.0 * (1.0 - 3.0 * ez2) * invR3 -
                                   2.0 * (1.0 - 5.0 * ez2) * invR5) /
                                 3.0;
            const double fact2 = -(3.0 * (1.0 - 6.0 * h_hat * t1) * invR -
                                   6.0 * (1.0 - 5.0 * ez2) * invR3 +
                                   10.0 * (1.0 - 7.0 * ez2) * invR5) /
                                 3.0;
            const double fact3 = ez *
                                 (3.0 * h_hat * (1.0 - 6.0 * t1) * invR -
                                  6.0 * (1.0 - 5.0 * ez2) * invR3 +
                                  10.0 * (2.0 - 7.0 * ez2) * invR5) *
                                 2.0 / 3.0;
            const double fact4 =
                ez * (3.0 * h_hat * invR - 10.0 * invR5) * 2.0 / 3.0;
            const double fact5 =
                -(3.0 * h_hat * h_hat * ez2 * invR + 3.0 * ez2 * invR3 +
                  (2.0 - 15.0 * ez2) * invR5) *
                4.0 / 3.0;

            Mxx += fact1 + fact2 * ex * ex;
            Mxy += fact2 * ex * ey;
            Mxz += fact2 * ex * ez + fact3 * ex;
            Myx += fact2 * ey * ex;
            Myy += fact1 + fact2 * ey * ey;
            Myz += fact2 * ey * ez + fact3 * ey;
            Mzx += fact2 * ez * ex + fact4 * ex;
            Mzy += fact2 * ez * ey + fact4 * ey;
            Mzz += fact1 + fact2 * ez2 + fact3 * ez + fact4 * ez + fact5;

            u[i * 3 + 0] +=
                (Mxx * force(j, 0) + Mxy * force(j, 1) + Mxz * force(j, 2)) *
                norm_fact_f;
            u[i * 3 + 1] +=
                (Myx * force(j, 0) + Myy * force(j, 1) + Myz * force(j, 2)) *
                norm_fact_f;
            u[i * 3 + 2] +=
                (Mzx * force(j, 0) + Mzy * force(j, 1) + Mzz * force(j, 2)) *
                norm_fact_f;
            u[j * 3 + 0] +=
                (Mxx * force(i, 0) + Myx * force(i, 1) + Mzx * force(i, 2)) *
                norm_fact_f;
            u[j * 3 + 1] +=
                (Mxy * force(i, 0) + Myy * force(i, 1) + Mzy * force(i, 2)) *
                norm_fact_f;
            u[j * 3 + 2] +=
                (Mxz * force(i, 0) + Myz * force(i, 1) + Mzz * force(i, 2)) *
                norm_fact_f;
        }
    }

    return u;
}

#ifdef PYTHON
PYBIND11_MODULE(mobility_cpp, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("single_wall_mobility_trans_times_force",
          &single_wall_mobility_trans_times_force, "Calculate M*f");
    m.def("single_wall_fluid_mobility", &single_wall_fluid_mobility, "");
    m.def("damping_matrix_B", &damping_matrix_B, "");
    m.def("rotne_prager_tensor", &rotne_prager_tensor, "Rotne-Prager tensor.");
}
#endif
