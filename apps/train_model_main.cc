#include "core/data_converter.h"
#include "core/data_processor.h"
#include "core/written_number.h"

using namespace naivebayes;

int main() {
  std::ifstream input_file("/Users/lirunfeng/cinder_master/my-projects/"
                           "naive-bayes-ranrandy/data/trainingimagesandlabels"
                           ".txt");
  if (input_file.is_open()) {
    DataConverter data_converter;
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
