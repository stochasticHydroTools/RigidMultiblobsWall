#include <iostream>
#include <mobility/mobility.hpp>
#include <multi_bodies/forces.hpp>

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

    fclose(fin);

    return {rvec, force, eta, a, L};
}

int main(int argc, char *argv[]) {
    auto [rvec, force, eta, a, L] = load_data("input.bin");

    double start = omp_get_wtime();
    dvec res = single_wall_mobility_trans_times_force(rvec, force, eta, a, L);
    std::cout << omp_get_wtime() - start << std::endl;

    return 0;
}
