#include "core/data_converter.h"
#include "core/data_processor.h"
#include "core/written_number.h"

using namespace naivebayes;

int main(int argc, char* argv[]) {
  // argv[1] = load_model_path
  if (argc == 2) {
    std::ifstream input_file(argv[1]);
    
    if (input_file.is_open()) {
      DataProcessor data_processor;
      input_file >> data_processor;
      input_file.close();
    }
  }

  // argv[1] = laplace parameter
  // argv[2] = dataset_path
  // argv[3] = save_model_path
  if (argc == 4) {
    std::ifstream input_file(argv[2]);

    if (input_file.is_open()) {
      DataConverter data_converter;
      input_file >> data_converter;

      DataProcessor data_processor(data_converter, std::stod(argv[1]));
      std::ofstream output_file(argv[3]);
      output_file << data_processor;

      output_file.close();
      input_file.close();
    }
  }
  return 0;
}
