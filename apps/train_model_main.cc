#include "core/data_converter.h"
#include "core/data_processor.h"
#include "core/written_number.h"

using namespace naivebayes;

// TODO: You may want to change main's signature to take in argc and argv
int main() {
  // TODO: Replace this with code that reads the training data, trains a model,
  // and saves the trained model to a file.
  std::ifstream input_file("/Users/lirunfeng/cinder_master/my-projects/"
                           "naive-bayes-ranrandy/data/trainingimagesandlabels"
                           ".txt");
  DataConverter data_converter;
  if (input_file.is_open()) {
    input_file >> data_converter;
    DataProcessor data_processor(data_converter, 100);
    std::ofstream output_file("/Users/lirunfeng/cinder_master/my-projects/"
                              "naive-bayes-ranrandy/data/computed_probabilities"
                              ".txt");
    output_file << data_processor;
    
    input_file.close();
  }
  return 0;
}
