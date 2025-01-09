#include "../include/simd_engine.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char* argv[]) {
    try {
        std::string weight_file = "weight_bits.bin";  
        
        // Parse command line args
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
                weight_file = argv[i + 1];
                i++;  
            }
        }

        // Check file existence
        if (!std::filesystem::exists(weight_file)) {
            std::string parent_path = "../" + weight_file;
            if (std::filesystem::exists(parent_path)) {
                weight_file = parent_path;
            } else {
                throw std::runtime_error("Weight File Not Found: " + weight_file);
            }
        }

        std::cout << "Loading weights from: " << weight_file << std::endl;
        mpX::SIMDEngine engine(weight_file);
        
        // Test computation
        std::vector<int16_t> activations = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        };
        
        mpX::Tile<int32_t> result = engine.compute(activations);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}