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
              << "  -t, --hopping     Hopping parameter\n"
              << "  -U, --interaction On-site interaction\n"
              << "  -u, --potential   Chemical potential\n"
              << "  -r, --range       Range for varying parameters (if range is the same for each)\n"
              << "  -s, --step        Step for varying parameters (with s < r)\n"
              << "  -f, --fixed       Fixed parameter (t, U or u)\n"
              << "  -S, --scale       Spacing scale: 'lin' (linear) or 'log' (logarithmic, default)\n"
              << "  -T, --sigma-t     Disorder variance for hopping (default: 0.0)\n"
              << "  -V, --delta-U     Disorder half-width for interaction (default: 0.0)\n"
              << "  -v, --delta-u     Disorder half-width for chemical potential (default: 0.0)\n"
              << "  -R, --realizations Number of disorder realizations (default: 1)\n";
}


int main(int argc, char *argv[]) {

    // PARAMETERS OF THE MODEL
    int m, n;
    double t, U, mu, s, r;
    double sigma_t = 0.0, delta_U = 0.0, delta_u = 0.0;
    int realizations = 1;
    std::string fixed_param;
    std::string scale = "log";

    const char* const short_opts = "m:n:t:U:u:r:s:f:S:T:V:v:R:h";
    const option long_opts[] = {
        {"sites", required_argument, nullptr, 'm'},
        {"bosons", required_argument, nullptr, 'n'},
        {"hopping", required_argument, nullptr, 't'},
        {"interaction", required_argument, nullptr, 'U'},
        {"potential", required_argument, nullptr, 'u'},
        {"range", required_argument, nullptr, 'r'},
        {"step", required_argument, nullptr, 's'},
        {"fixed", required_argument, nullptr, 'f'},
        {"scale", required_argument, nullptr, 'S'},
        {"sigma-t", required_argument, nullptr, 'T'},
        {"delta-U", required_argument, nullptr, 'V'},
        {"delta-u", required_argument, nullptr, 'v'},
        {"realizations", required_argument, nullptr, 'R'},
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
            case 't':
                t = std::stod(optarg);
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
            case 'S':
                scale = optarg;
                break;
            case 'T':
                sigma_t = std::stod(optarg);
                break;
            case 'V':
                delta_U = std::stod(optarg);
                break;
            case 'v':
                delta_u = std::stod(optarg);
                break;
            case 'R':
                realizations = std::stoi(optarg);
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
    if(fixed_param != "t" && fixed_param != "U" && fixed_param != "u"){
        std::cerr << "Error: fixed parameter must be t, U or u." << std::endl;
        return 1;
    }
    if (realizations < 1) {
        std::cerr << "Error: realizations must be at least 1." << std::endl;
        return 1;
    }
    if (scale != "lin" && scale != "log") {
        std::cerr << "Error: scale must be 'lin' or 'log'." << std::endl;
        return 1;
    }

    // Calculate the exact parameters
    Analysis::exact_parameters(m, n, t, U, mu, s, r, fixed_param, sigma_t, delta_U, delta_u, realizations, scale);

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