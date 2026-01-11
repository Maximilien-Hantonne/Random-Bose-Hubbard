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
              << "Options:\n"
              << "  -m, --sites       Number of sites (single, list, or range)\n"
              << "  -n, --bosons      Number of bosons (single, list, or range)\n"
              << "  -t, --hopping     Hopping parameter initial value (single, list, or range)\n"
              << "  -U, --interaction On-site interaction initial value (single, list, or range)\n"
              << "  -u, --potential   Chemical potential initial value (single, list, or range)\n"
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
              << "  -R, --realizations Number of disorder realizations (default: 1)\n"
              << "\n"
              << "  Parameter formats:\n"
              << "    Single value:  -v 0.3\n"
              << "    Comma list:    -v 0.3,0.5,0.7\n"
              << "    Range:         -v 0.1:0.2:0.9  (start:step:end)\n"
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

    // copyright();

    // PARAMETERS OF THE MODEL - now stored as vectors
    std::vector<int> m_list, n_list;
    std::vector<double> t_list, U_list, mu_list;
    std::vector<double> r_t_list, r_U_list, r_u_list;
    std::vector<double> s_t_list, s_U_list, s_u_list;
    std::vector<double> delta_t_list, delta_U_list, delta_u_list;
    std::vector<int> realizations_list;
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
            case 'a':
                r_t_list = parse_param_list(optarg);
                break;
            case 'b':
                r_U_list = parse_param_list(optarg);
                break;
            case 'c':
                r_u_list = parse_param_list(optarg);
                break;
            case 'A':
                s_t_list = parse_param_list(optarg);
                break;
            case 'B':
                s_U_list = parse_param_list(optarg);
                break;
            case 'C':
                s_u_list = parse_param_list(optarg);
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
                delta_t_list = parse_param_list(optarg);
                break;
            case 'V':
                delta_U_list = parse_param_list(optarg);
                break;
            case 'v':
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
        std::cerr << "Error: hopping parameter (-t) must be specified." << std::endl;
        return 1;
    }
    if (U_list.empty()) {
        std::cerr << "Error: interaction parameter (-U) must be specified." << std::endl;
        return 1;
    }
    if (mu_list.empty()) {
        std::cerr << "Error: chemical potential (-u) must be specified." << std::endl;
        return 1;
    }
    
    // Validate ranges and steps for non-fixed parameters (check first element)
    if (fixed_param != "t" && r_t_list[0] <= 0.0) {
        std::cerr << "Error: range for t (-a) must be specified and positive when t is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "U" && r_U_list[0] <= 0.0) {
        std::cerr << "Error: range for U (-b) must be specified and positive when U is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "u" && r_u_list[0] <= 0.0) {
        std::cerr << "Error: range for u (-c) must be specified and positive when u is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "t" && s_t_list[0] <= 0.0) {
        std::cerr << "Error: step for t (-A) must be specified and positive when t is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "U" && s_U_list[0] <= 0.0) {
        std::cerr << "Error: step for U (-B) must be specified and positive when U is not fixed." << std::endl;
        return 1;
    }
    if (fixed_param != "u" && s_u_list[0] <= 0.0) {
        std::cerr << "Error: step for u (-C) must be specified and positive when u is not fixed." << std::endl;
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
    if (distrib != "uni" && distrib != "gaus") {
        std::cerr << "Error: distrib must be 'uni' or 'gaus'." << std::endl;
        return 1;
    }

    // Helper lambda to run python plotting
    auto run_python_script = []() -> int {
        return system("python3 plot.py");
    };

    // Calculate total runs and counter
    size_t total_runs = m_list.size() * n_list.size() * t_list.size() * U_list.size() * 
                        mu_list.size() * r_t_list.size() * r_U_list.size() * r_u_list.size() *
                        s_t_list.size() * s_U_list.size() * s_u_list.size() * 
                        delta_t_list.size() * delta_U_list.size() * delta_u_list.size() *
                        realizations_list.size();
    size_t run_counter = 0;

    // Iterate over all parameter combinations
    for (int m : m_list) {
    for (int n : n_list) {
    for (double t : t_list) {
    for (double U : U_list) {
    for (double mu : mu_list) {
    for (double r_t : r_t_list) {
    for (double r_U : r_U_list) {
    for (double r_u : r_u_list) {
    for (double s_t : s_t_list) {
    for (double s_U : s_U_list) {
    for (double s_u : s_u_list) {
    for (double delta_t : delta_t_list) {
    for (double delta_U : delta_U_list) {
    for (double delta_u : delta_u_list) {
    for (int realizations : realizations_list) {
        run_counter++;
        std::cout << "\n Run " << run_counter << "/" << total_runs << "with parameters: m=" << m << ", n=" << n << ", t=" << t << ", U=" << U 
                  << ", mu=" << mu << ", delta_t=" << delta_t << ", delta_U=" << delta_U 
                  << ", delta_u=" << delta_u << ", R=" << realizations << std::endl;

        // Calculate the exact parameters
        Analysis::exact_parameters(m, n, t, U, mu, s_t, s_U, s_u, r_t, r_U, r_u, fixed_param, delta_t, delta_U, delta_u, realizations, scale, distrib);

        // Execute the Python script to plot the results
        std::future<int> result = std::async(std::launch::async, run_python_script);
        if (result.get() != 0) {
            std::cerr << "Error when executing Python script for run " << run_counter << "." << std::endl;
            // Continue to next iteration instead of returning
        }
    }}}}}}}}}}}}}}}

    std::cout << "\nAll " << total_runs << " runs finished !" << std::endl;

    return 0;
}