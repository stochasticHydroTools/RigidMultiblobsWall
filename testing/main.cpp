#include <cmath>
#include <iostream>

#include <cstdio>

#include <Eigen/Core>

#include <omp.h>

// Some convenience types
typedef Eigen::ArrayXd dvec;
typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    dvecvec;

#pragma omp declare reduction (+: dvec: omp_out=omp_out+omp_in)\
     initializer(omp_priv=dvec::Zero(omp_orig.size()))

dvec single_wall_mobility_trans_times_force(const dvecvec &r_vectors,
                                            const dvecvec &force, double eta,
                                            double a, const dvec &L) {
    const int N = r_vectors.rows();
    dvec u = dvec(N * 3).setZero();
    const double fourOverThree = 4.0 / 3.0;
    const double inva = 1.0 / a;
    const double norm_fact_f = 1.0 / (8.0 * M_PI * eta * a);

    // Loop over image boxes and then over particles
#pragma omp parallel for reduction(+ : u) schedule(runtime)
    for (int i = 0; i < N; ++i) {
        //     for boxX in range(-periodic_x, periodic_x+1):
        //       for boxY in range(-periodic_y, periodic_y+1):
        //         for boxZ in range(-periodic_z, periodic_z+1):
        for (int j = i; j < N; ++j) {
            double Mxx, Mxy, Mxz;
            double Myx, Myy, Myz;
            double Mzx, Mzy, Mzz;

            // Compute scaled vector between particles i and j
            Eigen::Vector3d dr = inva * (r_vectors.row(i) - r_vectors.row(j));

            int j_image = j;
            if (i == j_image) {
                Mxx = fourOverThree;
                Mxy = 0;
                Mxz = 0;
                Myy = Mxx;
                Myz = 0;
                Mzz = Mxx;
            } else {
                // Normalize distance with hydrodynamic radius
                const double r2 = dr.squaredNorm();
                const double r = sqrt(r2);

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
            }
            Myx = Mxy;
            Mzx = Mxz;
            Mzy = Myz;

            // Wall correction
            dr[2] = (r_vectors(i, 2) + r_vectors(j, 2)) / a;
            double hj = r_vectors(j, 2) / a;

            if (i == j_image) {
                const double invZi = 1.0 / hj;
                const double invZi3 = invZi * invZi * invZi;
                const double invZi5 = invZi3 * invZi * invZi;

                Mxx += -(9.0 * invZi - 2.0 * invZi3 + invZi5) / 12.0;
                Myy += -(9.0 * invZi - 2.0 * invZi3 + invZi5) / 12.0;
                Mzz += -(9.0 * invZi - 4.0 * invZi3 + invZi5) / 6.0;
            } else {
                const double h_hat = hj / dr[2];
                const double invR = 1.0 / dr.norm();
                const double ex = dr[0] * invR;
                const double ey = dr[1] * invR;
                const double ez = dr[2] * invR;
                const double invR3 = invR * invR * invR;
                const double invR5 = invR3 * invR * invR;

                const double fact1 =
                    -(3.0 * (1.0 + 2.0 * h_hat * (1.0 - h_hat) * ez * ez) *
                          invR +
                      2.0 * (1.0 - 3.0 * ez * ez) * invR3 -
                      2.0 * (1.0 - 5.0 * ez * ez) * invR5) /
                    3.0;
                const double fact2 =
                    -(3.0 * (1.0 - 6.0 * h_hat * (1.0 - h_hat) * ez * ez) *
                          invR -
                      6.0 * (1.0 - 5.0 * ez * ez) * invR3 +
                      10.0 * (1.0 - 7.0 * ez * ez) * invR5) /
                    3.0;
                const double fact3 =
                    ez *
                    (3.0 * h_hat * (1.0 - 6.0 * (1.0 - h_hat) * ez * ez) *
                         invR -
                     6.0 * (1.0 - 5.0 * ez * ez) * invR3 +
                     10.0 * (2.0 - 7.0 * ez * ez) * invR5) *
                    2.0 / 3.0;
                const double fact4 =
                    ez * (3.0 * h_hat * invR - 10.0 * invR5) * 2.0 / 3.0;
                const double fact5 =
                    -(3.0 * h_hat * h_hat * ez * ez * invR +
                      3.0 * ez * ez * invR3 + (2.0 - 15.0 * ez * ez) * invR5) *
                    4.0 / 3.0;

                Mxx += fact1 + fact2 * ex * ex;
                Mxy += fact2 * ex * ey;
                Mxz += fact2 * ex * ez + fact3 * ex;
                Myx += fact2 * ey * ex;
                Myy += fact1 + fact2 * ey * ey;
                Myz += fact2 * ey * ez + fact3 * ey;
                Mzx += fact2 * ez * ex + fact4 * ex;
                Mzy += fact2 * ez * ey + fact4 * ey;
                Mzz +=
                    fact1 + fact2 * ez * ez + fact3 * ez + fact4 * ez + fact5;
            }

            u[i * 3 + 0] +=
                (Mxx * force(j, 0) + Mxy * force(j, 1) + Mxz * force(j, 2)) *
                norm_fact_f;
            u[i * 3 + 1] +=
                (Myx * force(j, 0) + Myy * force(j, 1) + Myz * force(j, 2)) *
                norm_fact_f;
            u[i * 3 + 2] +=
                (Mzx * force(j, 0) + Mzy * force(j, 1) + Mzz * force(j, 2)) *
                norm_fact_f;

            if (i != j) {
                u[j * 3 + 0] += (Mxx * force(i, 0) + Myx * force(i, 1) +
                                 Mzx * force(i, 2)) *
                                norm_fact_f;
                u[j * 3 + 1] += (Mxy * force(i, 0) + Myy * force(i, 1) +
                                 Mzy * force(i, 2)) *
                                norm_fact_f;
                u[j * 3 + 2] += (Mxz * force(i, 0) + Myz * force(i, 1) +
                                 Mzz * force(i, 2)) *
                                norm_fact_f;
            }
        }
    }

    return u;
}

std::tuple<dvecvec, dvecvec, double, double, dvec>
load_data(std::string filename) {
    int64_t N;

    FILE *fin = fopen(filename.c_str(), "r");

    fread(&N, sizeof(int64_t), 1, fin);
    double eta;
    double a;

    std::vector<double> buffer(N * 3);
    fread(buffer.data(), sizeof(double), N * 3, fin);
    dvecvec rvec = Eigen::Map<dvecvec>(buffer.data(), N, 3);
    fread(buffer.data(), sizeof(double), N * 3, fin);
    dvecvec force = Eigen::Map<dvecvec>(buffer.data(), N, 3);
    fread(&eta, sizeof(double), 1, fin);
    fread(&a, sizeof(double), 1, fin);
    fread(buffer.data(), sizeof(double), 3, fin);
    dvec L = Eigen::Map<dvec>(buffer.data(), 3);

    return {rvec, force, eta, a, L};
}

int main(int argc, char *argv[]) {
    auto [rvec, force, eta, a, L] = load_data("input.bin");

    double start = omp_get_wtime();
    dvec res = single_wall_mobility_trans_times_force(rvec, force, eta, a, L);
    std::cout << omp_get_wtime() - start << std::endl;

    return 0;
}
