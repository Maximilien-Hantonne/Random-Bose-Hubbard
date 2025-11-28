#include <cmath>
#include <future>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>

#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <getopt.h>
#include <omp.h>

#include "analysis.hpp"

void print_usage() {
    std::cout << "Usage: program [options]\n"
              << "Options:\n"
              << "  -m, --sites       Number of sites\n"
              << "  -n, --bosons      Number of bosons\n"
              << "  -J, --hopping     Hopping parameter\n"
              << "  -U, --interaction On-site interaction\n"
              << "  -u, --potential   Chemical potential\n"
              << "  -r, --range     Range for varying parameters (if range is the same for each)\n"
              << "  -s, --step      Step for varying parameters (with s < r)\n"
              << "  -f --fixed      Fixed parameter (J, U or u) \n";
}


int main(int argc, char *argv[]) {

    // PARAMETERS OF THE MODEL
    int m, n;
    double J, U, mu, s, r;
    std::string fixed_param;

    const char* const short_opts = "m:n:J:U:u:r:s:f:h";
    const option long_opts[] = {
        {"sites", required_argument, nullptr, 'm'},
        {"bosons", required_argument, nullptr, 'n'},
        {"hopping", required_argument, nullptr, 'J'},
        {"interaction", required_argument, nullptr, 'U'},
        {"potential", required_argument, nullptr, 'u'},
        {"range", required_argument, nullptr, 'r'},
        {"step", required_argument, nullptr, 's'},
        {"fixed", required_argument, nullptr, 'f'},
		{"help", no_argument, nullptr, 'h'},
        {nullptr, no_argument, nullptr, 0}
    };

    while (true) {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
        if (-1 == opt) break;
        switch (opt) {
            case 'm':
                m = std::stoi(optarg);
                break;
            case 'n':
                n = std::stoi(optarg);
                break;
            case 'J':
                J = std::stod(optarg);
                break;
            case 'U':
                U = std::stod(optarg);
                break;
            case 'u':
                mu = std::stod(optarg);
                break;
            case 'r':
                r = std::stod(optarg);
                break;
            case 's':
                s = std::stod(optarg);
                break;
            case 'f':
                fixed_param = optarg;
                break;
            case 'h':
            default:
                print_usage();
                return 0;
        }
    }

    if (s >= r) {
        std::cerr << "Error: s must be smaller than r." << std::endl;
        return 1;
    }
    if(fixed_param != "J" && fixed_param != "U" && fixed_param != "u"){
        std::cerr << "Error: fixed parameter must be J, U or u." << std::endl;
        return 1;
    }

    // Calculate the exact parameters
    Analysis::exact_parameters(m, n, J, U, mu, s, r, fixed_param);

    // Execute the Python script to plot the results
    auto run_python_script = []() -> int {
        return system("python3 plot.py");
    };
    std::future<int> result = std::async(std::launch::async, run_python_script);
    if (result.get() != 0) {
        std::cerr << "Error when executing Python script." << std::endl;
        return 1;
    }
}