#include "config.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <nlohmann/json.hpp>  


//    █████████                         ██████   ███          
//   ███░░░░░███                       ███░░███ ░░░           
//  ███     ░░░   ██████  ████████    ░███ ░░░  ████   ███████
// ░███          ███░░███░░███░░███  ███████   ░░███  ███░░███
// ░███         ░███ ░███ ░███ ░███ ░░░███░     ░███ ░███ ░███
// ░░███     ███░███ ░███ ░███ ░███   ░███      ░███ ░███ ░███
//  ░░█████████ ░░██████  ████ █████  █████     █████░░███████
//   ░░░░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░     ░░░░░  ░░░░░███
//                                                    ███ ░███
//                                                   ░░██████ 
//                                                    ░░░░░░  


namespace perf_model {


namespace config {


using json = nlohmann::json;
int num_pes = 8;
int num_matmuls = 1;                        // Default to 1 Matrix Multiplication
double expected_weight_sparsity = 0.0;      // Default to no Sparsity
double expected_activation_sparsity = 0.0;  // Default to no Sparsity
int adder_tree_width = 4;
int accumulation_mode = 0;                  // Default to Row-Wise (0)
int synchronization_mode = 0;               // Default to Global Stalling (0)
int batch_size = 8;                         // Default Batch Size for Synchronisation Mode 2


void loadConfig(const std::string &filename) {
    std::ifstream f(filename);
    if (!f.good()) {
        std::cerr << "Config file " << filename << " does not exist, using defaults."
                  << std::endl;
        return;
    }
    nlohmann::json config;
    f >> config;

    if (config.contains("num_pes")) {
        num_pes = config["num_pes"];
    }

    if (config.contains("num_matmuls")) {
        num_matmuls = config["num_matmuls"];
    }

    if (config.contains("expected_weight_sparsity")) {
        expected_weight_sparsity = config["expected_weight_sparsity"];
    }

    if (config.contains("expected_activation_sparsity")) {
        expected_activation_sparsity = config["expected_activation_sparsity"];
    }
    
    if (config.contains("adder_tree_width")) {
        adder_tree_width = config["adder_tree_width"];
    }
    
    if (config.contains("accumulation_mode")) {
        if (config["accumulation_mode"].is_string()) {
            std::string mode = config["accumulation_mode"];
            if (mode == "row_wise") {
                accumulation_mode = 0;
            } else if (mode == "local_pe") {
                accumulation_mode = 1;
            } else {
                std::cerr << "Unknown accumulation mode: " << mode << ", using default." << std::endl;
            }
        } else {
            accumulation_mode = config["accumulation_mode"];
        }
    }
    
    if (config.contains("synchronization_mode")) {
        synchronization_mode = config["synchronization_mode"];
    }
    
    if (config.contains("batch_size")) {
        batch_size = config["batch_size"];
    }
}


void saveConfig(const std::string &filename) {
    nlohmann::json config;
    config["num_pes"] = num_pes;
    config["num_matmuls"] = num_matmuls;
    config["expected_weight_sparsity"] = expected_weight_sparsity;
    config["expected_activation_sparsity"] = expected_activation_sparsity;
    config["adder_tree_width"] = adder_tree_width;
    config["accumulation_mode"] = accumulation_mode;
    config["synchronization_mode"] = synchronization_mode;
    config["batch_size"] = batch_size;

    std::ofstream f(filename);
    f << config.dump(4);
}


} // namespace config
} // namespace perf_model 