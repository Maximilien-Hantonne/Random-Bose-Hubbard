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
              << "  -t, --hopping     Hopping parameter (initial value)\n"
              << "  -U, --interaction On-site interaction (initial value)\n"
              << "  -u, --potential   Chemical potential (initial value)\n"
              << "  -a, --range-t     Range for hopping parameter t\n"
              << "  -b, --range-U     Range for interaction parameter U\n"
              << "  -c, --range-u     Range for chemical potential u\n"
              << "  -A, --step-t      Step for hopping parameter t\n"
              << "  -B, --step-U      Step for interaction parameter U\n"
              << "  -C, --step-u      Step for chemical potential u\n"
              << "  -f, --fixed       Fixed parameter (t, U or u)\n"
              << "  -S, --scale       Spacing scale: 'lin' (linear) or 'log' (logarithmic, default)\n"
              << "  -d, --distrib     Disorder distribution: 'uni' (uniform, default) or 'gaus' (gaussian)\n"
              << "  -T, --delta-t     Disorder half-width for hopping (default: 0.0)\n"
              << "  -V, --delta-U     Disorder half-width for interaction (default: 0.0)\n"
              << "  -v, --delta-u     Disorder half-width for chemical potential (default: 0.0)\n"
              << "  -R, --realizations Number of disorder realizations (default: 1)\n";
}


int main(int argc, char *argv[]) {

    // PARAMETERS OF THE MODEL
    int m, n;
    double t, U, mu;
    double r_t = 0.0, r_U = 0.0, r_u = 0.0;  // Separate ranges for each parameter
    double s_t = 0.0, s_U = 0.0, s_u = 0.0;  // Separate steps for each parameter
    double delta_t = 0.0, delta_U = 0.0, delta_u = 0.0;
    int realizations = 1;
    std::string fixed_param;
    std::string scale = "log";
    std::string distrib = "uni";

    const char* const short_opts = "m:n:t:U:u:a:b:c:A:B:C:f:S:d:T:V:v:R:h";
    const option long_opts[] = {
        {"sites", required_argument, nullptr, 'm'},
        {"bosons", required_argument, nullptr, 'n'},
        {"hopping", required_argument, nullptr, 't'},
        {"interaction", required_argument, nullptr, 'U'},
        {"potential", required_argument, nullptr, 'u'},
        {"range-t", required_argument, nullptr, 'a'},
        {"range-U", required_argument, nullptr, 'b'},
        {"range-u", required_argument, nullptr, 'c'},
        {"step-t", required_argument, nullptr, 'A'},
        {"step-U", required_argument, nullptr, 'B'},
        {"step-u", required_argument, nullptr, 'C'},
        {"fixed", required_argument, nullptr, 'f'},
        {"scale", required_argument, nullptr, 'S'},
        {"distrib", required_argument, nullptr, 'd'},
        {"delta-t", required_argument, nullptr, 'T'},
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
            case 'a':
                r_t = std::stod(optarg);
                break;
            case 'b':
                r_U = std::stod(optarg);
                break;
            case 'c':
                r_u = std::stod(optarg);
                break;
            case 'A':
                s_t = std::stod(optarg);
                break;
            case 'B':
                s_U = std::stod(optarg);
                break;
            case 'C':
                s_u = std::stod(optarg);
                break;
            case 'f':
                fixed_param = optarg;
                break;
            case 'S':
                scale = optarg;
                break;
            case 'd':
                distrib = optarg;
                break;
            case 'T':
                delta_t = std::stod(optarg);
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

    if(fixed_param != "t" && fixed_param != "U" && fixed_param != "u"){
        std::cerr << "Error: fixed parameter must be t, U or u." << std::endl;
        return 1;
    }
    
    // Validate that non-fixed parameters have ranges and steps specified
    if (fixed_param != "t" && r_t <= 0.0) {
        std::cerr << "Error: range for t (-a) must be specified and positive when t is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "U" && r_U <= 0.0) {
        std::cerr << "Error: range for U (-b) must be specified and positive when U is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "u" && r_u <= 0.0) {
        std::cerr << "Error: range for u (-c) must be specified and positive when u is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "t" && s_t <= 0.0) {
        std::cerr << "Error: step for t (-A) must be specified and positive when t is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "U" && s_U <= 0.0) {
        std::cerr << "Error: step for U (-B) must be specified and positive when U is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "u" && s_u <= 0.0) {
        std::cerr << "Error: step for u (-C) must be specified and positive when u is not fixed." << std::endl;
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
    if (distrib != "uni" && distrib != "gaus") {
        std::cerr << "Error: distrib must be 'uni' or 'gaus'." << std::endl;
        return 1;
    }

    // Calculate the exact parameters
    Analysis::exact_parameters(m, n, t, U, mu, s_t, s_U, s_u, r_t, r_U, r_u, fixed_param, delta_t, delta_U, delta_u, realizations, scale, distrib);

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