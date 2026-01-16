#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <getopt.h>
#include <omp.h>

#include "analysis.hpp"

// Parse a parameter
std::vector<double> parse_param_list(const std::string& arg) {
    std::vector<double> values;
    std::string s = arg;
    
    // Trim whitespace
    s.erase(0, s.find_first_not_of(" \t"));
    s.erase(s.find_last_not_of(" \t") + 1);
    
    if (s.empty()) return values;
    
    // Check for range syntax (start:step:end)
    size_t colon_count = std::count(s.begin(), s.end(), ':');
    if (colon_count == 2) {
        // Range syntax: start:step:end
        size_t first_colon = s.find(':');
        size_t second_colon = s.find(':', first_colon + 1);
        
        double start = std::stod(s.substr(0, first_colon));
        double step = std::stod(s.substr(first_colon + 1, second_colon - first_colon - 1));
        double end = std::stod(s.substr(second_colon + 1));
        
        if (step > 0) {
            for (double v = start; v <= end + 1e-9; v += step) {
                values.push_back(v);
            }
        } else if (step < 0) {
            for (double v = start; v >= end - 1e-9; v += step) {
                values.push_back(v);
            }
        }
        return values;
    }
    
    // Comma-separated list or single value
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        if (!token.empty()) {
            values.push_back(std::stod(token));
        }
    }
    
    return values;
}

// Parse an integer parameter list (same formats as double)
std::vector<int> parse_int_param_list(const std::string& arg) {
    std::vector<int> values;
    std::string s = arg;
    
    s.erase(0, s.find_first_not_of(" \t"));
    s.erase(s.find_last_not_of(" \t") + 1);
    
    if (s.empty()) return values;
    
    // Check for range syntax (start:step:end)
    size_t colon_count = std::count(s.begin(), s.end(), ':');
    if (colon_count == 2) {
        size_t first_colon = s.find(':');
        size_t second_colon = s.find(':', first_colon + 1);
        
        int start = std::stoi(s.substr(0, first_colon));
        int step = std::stoi(s.substr(first_colon + 1, second_colon - first_colon - 1));
        int end = std::stoi(s.substr(second_colon + 1));
        
        if (step > 0) {
            for (int v = start; v <= end; v += step) {
                values.push_back(v);
            }
        } else if (step < 0) {
            for (int v = start; v >= end; v += step) {
                values.push_back(v);
            }
        }
        return values;
    }
    
    // Comma-separated list or single value
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        if (!token.empty()) {
            values.push_back(std::stoi(token));
        }
    }
    
    return values;
}



void print_usage() {
    std::cout << "Usage: program [options]\n"
              << "\nSystem size:\n"
              << "  -m, --sites        Number of sites\n"
              << "  -n, --bosons       Number of bosons\n"
              << "\nModel parameters:\n"
              << "  -t, --hopping            Hopping parameter\n"
              << "  -U, --interaction        On-site interaction\n"
              << "  -u, --chemical-potential Chemical potential\n"
              << "\nParameter scans:\n"
              << "      --tr  Range for hopping t\n"
              << "      --Ur  Range for interaction U\n"
              << "      --ur  Range for chemical potential mu\n"
              << "      --ts  Step for hopping t\n"
              << "      --Us  Step for interaction U\n"
              << "      --us  Step for chemical potential mu\n"
              << "\nDisorder:\n"
              << "      --tD  Disorder strength for t (default: 0.0)\n"
              << "      --UD  Disorder strength for U (default: 0.0)\n"
              << "      --uD  Disorder strength for mu (default: 0.0)\n"
              << "      --tp  Probability distribution for t: 'uni' or 'gaus' (default: uni)\n"
              << "      --Up  Probability distribution for U: 'uni' or 'gaus' (default: uni)\n"
              << "      --up  Probability distribution for mu: 'uni' or 'gaus' (default: uni)\n"
              << "  -R, --realizations Number of disorder realizations (default: 1)\n"
              << "\nOther options:\n"
              << "  -f, --fixed        Fixed parameter (t, U or u)\n"
              << "  -S, --scale        Spacing scale: 'lin' or 'log' (default: log)\n"
              << "  -h, --help         Display this help message\n"
              << "\n"
              << "Parameter formats:\n"
              << "  Single value:  -t 0.3\n"
              << "  Comma list:    -t 0.3,0.5,0.7\n"
              << "  Range:         -t 0.1:0.2:0.9  (start:step:end)\n"
              << "\n"
              << "List iteration: When multiple lists are provided, iterates by index\n"
              << "                (1st of each, 2nd of each, etc.), not all combinations.\n"
              << "                Shorter lists repeat their last value.\n"
              << "\n";
}

 void copyright() {
    std::cout << "======================================\n";
    std::cout << "Random Bose-Hubbard\n";
    std::cout << "Copyright (C) 2025 by Maximilien HANTONNE\n";
    std::cout << "This program is licensed under the GNU General Public License v3.0.\n";
    std::cout << "For more details, see the LICENSE file.\n";
    std::cout << "======================================\n\n";
}

int main(int argc, char *argv[]) {

    copyright();

    // PARAMETERS OF THE MODEL - now stored as vectors
    std::vector<int> m_list, n_list;
    std::vector<double> t_list, U_list, mu_list;
    std::vector<double> r_t_list, r_U_list, r_u_list;
    std::vector<double> s_t_list, s_U_list, s_u_list;
    std::vector<double> delta_t_list, delta_U_list, delta_u_list;
    std::vector<int> realizations_list;
    std::string fixed_param;
    std::string scale = "log";
    std::string distrib_t = "uni";  // Distribution for t (default: uniform)
    std::string distrib_U = "uni";  // Distribution for U (default: uniform)
    std::string distrib_u = "uni";  // Distribution for u (default: uniform)

    // Define option codes for long-only options
    enum {
        OPT_tr = 256, // --tr (range for t)
        OPT_Ur,       // --Ur (range for U)
        OPT_ur,       // --ur (range for u)
        OPT_ts,       // --ts (step for t)
        OPT_Us,       // --Us (step for U)
        OPT_us,       // --us (step for u)
        OPT_tD,       // --tD (disorder strength for t)
        OPT_UD,       // --UD (disorder strength for U)
        OPT_uD,       // --uD (disorder strength for u)
        OPT_tp,       // --tp (probability distribution for t)
        OPT_Up,       // --Up (probability distribution for U)
        OPT_up        // --up (probability distribution for u)
    };

    const char* const short_opts = "m:n:t:U:u:f:S:R:h";
    const option long_opts[] = {
        {"sites", required_argument, nullptr, 'm'},
        {"bosons", required_argument, nullptr, 'n'},
        {"hopping", required_argument, nullptr, 't'},
        {"interaction", required_argument, nullptr, 'U'},
        {"chemical-potential", required_argument, nullptr, 'u'},
        {"tr", required_argument, nullptr, OPT_tr},
        {"Ur", required_argument, nullptr, OPT_Ur},
        {"ur", required_argument, nullptr, OPT_ur},
        {"ts", required_argument, nullptr, OPT_ts},
        {"Us", required_argument, nullptr, OPT_Us},
        {"us", required_argument, nullptr, OPT_us},
        {"tD", required_argument, nullptr, OPT_tD},
        {"UD", required_argument, nullptr, OPT_UD},
        {"uD", required_argument, nullptr, OPT_uD},
        {"tp", required_argument, nullptr, OPT_tp},
        {"Up", required_argument, nullptr, OPT_Up},
        {"up", required_argument, nullptr, OPT_up},
        {"fixed", required_argument, nullptr, 'f'},
        {"scale", required_argument, nullptr, 'S'},
        {"realizations", required_argument, nullptr, 'R'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, no_argument, nullptr, 0}
    };

    while (true) {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
        if (-1 == opt) break;
        switch (opt) {
            case 'm':
                m_list = parse_int_param_list(optarg);
                break;
            case 'n':
                n_list = parse_int_param_list(optarg);
                break;
            case 't':
                t_list = parse_param_list(optarg);
                break;
            case 'U':
                U_list = parse_param_list(optarg);
                break;
            case 'u':
                mu_list = parse_param_list(optarg);
                break;
            case OPT_tr:
                r_t_list = parse_param_list(optarg);
                break;
            case OPT_Ur:
                r_U_list = parse_param_list(optarg);
                break;
            case OPT_ur:
                r_u_list = parse_param_list(optarg);
                break;
            case OPT_ts:
                s_t_list = parse_param_list(optarg);
                break;
            case OPT_Us:
                s_U_list = parse_param_list(optarg);
                break;
            case OPT_us:
                s_u_list = parse_param_list(optarg);
                break;
            case 'f':
                fixed_param = optarg;
                break;
            case 'S':
                scale = optarg;
                break;
            case OPT_tp:
                distrib_t = optarg;
                break;
            case OPT_Up:
                distrib_U = optarg;
                break;
            case OPT_up:
                distrib_u = optarg;
                break;
            case OPT_tD:
                delta_t_list = parse_param_list(optarg);
                break;
            case OPT_UD:
                delta_U_list = parse_param_list(optarg);
                break;
            case OPT_uD:
                delta_u_list = parse_param_list(optarg);
                break;
            case 'R':
                realizations_list = parse_int_param_list(optarg);
                break;
            case 'h':
            default:
                print_usage();
                return 0;
        }
    }

    // Set default values for empty lists
    if (r_t_list.empty()) r_t_list.push_back(0.0);
    if (r_U_list.empty()) r_U_list.push_back(0.0);
    if (r_u_list.empty()) r_u_list.push_back(0.0);
    if (s_t_list.empty()) s_t_list.push_back(0.0);
    if (s_U_list.empty()) s_U_list.push_back(0.0);
    if (s_u_list.empty()) s_u_list.push_back(0.0);
    if (delta_t_list.empty()) delta_t_list.push_back(0.0);
    if (delta_U_list.empty()) delta_U_list.push_back(0.0);
    if (delta_u_list.empty()) delta_u_list.push_back(0.0);
    if (realizations_list.empty()) realizations_list.push_back(1);

    if(fixed_param != "t" && fixed_param != "U" && fixed_param != "u"){
        std::cerr << "Error: fixed parameter must be t, U or u." << std::endl;
        return 1;
    }
    
    // Validate that required lists are not empty
    if (m_list.empty()) {
        std::cerr << "Error: number of sites (-m) must be specified." << std::endl;
        return 1;
    }
    if (n_list.empty()) {
        std::cerr << "Error: number of bosons (-n) must be specified." << std::endl;
        return 1;
    }
    if (t_list.empty()) {
        std::cerr << "Error: hopping parameter (--t) must be specified." << std::endl;
        return 1;
    }
    if (U_list.empty()) {
        std::cerr << "Error: interaction parameter (--U) must be specified." << std::endl;
        return 1;
    }
    if (mu_list.empty()) {
        std::cerr << "Error: chemical potential (--u) must be specified." << std::endl;
        return 1;
    }
    
    // Validate ranges and steps for non-fixed parameters (check first element)
    if (fixed_param != "t" && r_t_list[0] <= 0.0) {
        std::cerr << "Error: range for t (--tr) must be specified and positive when t is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "U" && r_U_list[0] <= 0.0) {
        std::cerr << "Error: range for U (--Ur) must be specified and positive when U is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "u" && r_u_list[0] <= 0.0) {
        std::cerr << "Error: range for u (--ur) must be specified and positive when u is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "t" && s_t_list[0] <= 0.0) {
        std::cerr << "Error: step for t (--ts) must be specified and positive when t is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "U" && s_U_list[0] <= 0.0) {
        std::cerr << "Error: step for U (--Us) must be specified and positive when U is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "u" && s_u_list[0] <= 0.0) {
        std::cerr << "Error: step for u (--us) must be specified and positive when u is not fixed." << std::endl;
        return 1;
    }
    for (int r : realizations_list) {
        if (r < 1) {
            std::cerr << "Error: realizations must be at least 1." << std::endl;
            return 1;
        }
    }
    if (scale != "lin" && scale != "log") {
        std::cerr << "Error: scale must be 'lin' or 'log'." << std::endl;
        return 1;
    }
    // Validate distribution values
    if (distrib_t != "uni" && distrib_t != "gaus") {
        std::cerr << "Error: --tp must be 'uni' or 'gaus', got '" << distrib_t << "'." << std::endl;
        return 1;
    }
    if (distrib_U != "uni" && distrib_U != "gaus") {
        std::cerr << "Error: --Up must be 'uni' or 'gaus', got '" << distrib_U << "'." << std::endl;
        return 1;
    }
    if (distrib_u != "uni" && distrib_u != "gaus") {
        std::cerr << "Error: --up must be 'uni' or 'gaus', got '" << distrib_u << "'." << std::endl;
        return 1;
    }

    // Helper lambda to run python plotting
    auto run_python_script = []() -> int {
        return system("python3 plot.py");
    };

    // Helper lambda to get value at index, or last value if index is out of bounds
    auto get_or_last_double = [](const std::vector<double>& vec, size_t idx) -> double {
        return vec[std::min(idx, vec.size() - 1)];
    };
    auto get_or_last_int = [](const std::vector<int>& vec, size_t idx) -> int {
        return vec[std::min(idx, vec.size() - 1)];
    };

    // Calculate total runs as maximum list size (index-based iteration)
    size_t total_runs = std::max({m_list.size(), n_list.size(), t_list.size(), U_list.size(), 
                        mu_list.size(), r_t_list.size(), r_U_list.size(), r_u_list.size(),
                        s_t_list.size(), s_U_list.size(), s_u_list.size(), 
                        delta_t_list.size(), delta_U_list.size(), delta_u_list.size(),
                        realizations_list.size()});

    // Iterate by index (not all combinations)
    for (size_t run_idx = 0; run_idx < total_runs; ++run_idx) {
        int m = get_or_last_int(m_list, run_idx);
        int n = get_or_last_int(n_list, run_idx);
        double t = get_or_last_double(t_list, run_idx);
        double U = get_or_last_double(U_list, run_idx);
        double mu = get_or_last_double(mu_list, run_idx);
        double r_t = get_or_last_double(r_t_list, run_idx);
        double r_U = get_or_last_double(r_U_list, run_idx);
        double r_u = get_or_last_double(r_u_list, run_idx);
        double s_t = get_or_last_double(s_t_list, run_idx);
        double s_U = get_or_last_double(s_U_list, run_idx);
        double s_u = get_or_last_double(s_u_list, run_idx);
        double delta_t = get_or_last_double(delta_t_list, run_idx);
        double delta_U = get_or_last_double(delta_U_list, run_idx);
        double delta_u = get_or_last_double(delta_u_list, run_idx);
        int realizations = get_or_last_int(realizations_list, run_idx);

        // Build progress prefix for this run
        std::string run_prefix = "Run " + std::to_string(run_idx + 1) + "/" + std::to_string(total_runs) + " - ";
        std::cout << "\n" << run_prefix << std::flush;

        // Exact calculations
        Analysis::exact_parameters(m, n, t, U, mu, s_t, s_U, s_u, r_t, r_U, r_u, fixed_param, delta_t, delta_U, delta_u, realizations, scale, distrib_t, distrib_U, distrib_u, run_prefix);

        // Execute the Python script to plot the results
        std::future<int> result = std::async(std::launch::async, run_python_script);
        if (result.get() != 0) {
            std::cerr << "Error when executing Python script for run " << (run_idx + 1) << "." << std::endl;
            // Continue to next iteration instead of returning
        }
    }

    std::cout << "\nAll runs finished " << std::endl;

    return 0;
}