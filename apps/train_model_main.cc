#include "core/data_converter.h"
#include "core/data_processor.h"
#include "core/written_number.h"

using namespace naivebayes;

int main(int argc, char* argv[]) {
  // argv[1] = load_model_path
  // argv[2] = test_file_path
  if (argc == 3) {
    std::ifstream input_file(argv[1]);
    std::ifstream test_input_file(argv[2]);
    
    if (input_file.is_open() && test_input_file.is_open()) {
      DataProcessor data_processor;
      input_file >> data_processor;
      
      DataConverter data_converter;
      test_input_file >> data_converter;
      
      std::vector<size_t> classification_result;
      std::vector<size_t> testing_results;
      
      for (const WrittenNumber& written_number : data_converter.GetDataset()) {
        testing_results.push_back(written_number.GetImageClass());
        std::map<size_t , double> likelihood_scores;
        size_t result = -1;
        double result_probability = log(0);

        for (auto & it : data_processor.GetClassProbability()) {
          double score = log(it.second);

          for (size_t i = 0; i < data_converter.GetImageSize(); i++) {
            for (size_t j = 0; j < data_converter.GetImageSize(); j++) {
              WrittenNumber::PixelColor pixelColor =
                  written_number.GetImageVector()[i][j];
              score += log(data_processor
                               .GetPixelProbability()[i][j][static_cast<size_t>(
                                   pixelColor)][it.first]);
            }
          }
          likelihood_scores[it.first] = score;
        }

        for (auto & it : likelihood_scores) {
          if (it.second > result_probability) {
            result = it.first;
            result_probability = it.second;
          }
        }
        
        classification_result.push_back(result);
      }
      
      size_t correct_result_count = 0;
      
      for (size_t i = 0; i < testing_results.size(); i++) {
        if (testing_results[i] == classification_result[i]) {
          correct_result_count++;
        }
      }
      
      std::cout << double(correct_result_count) / 
                       double(testing_results.size()) << std::endl;
      
      test_input_file.close();
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
