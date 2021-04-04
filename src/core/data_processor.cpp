#include "core/data_processor.h"

namespace naivebayes {

DataProcessor::DataProcessor(
    const std::vector<WrittenNumber>& written_numbers) {
  CalculateProbabilityForClasses(written_numbers);
}

const std::map<std::string, double>&
    DataProcessor::GetClassProbability() const {
  return image_class_probabilities_;
}

void DataProcessor::CalculateProbabilityForClasses(
    const std::vector<WrittenNumber>& written_numbers) {
  for (WrittenNumber written_number : written_numbers) {
    image_class_probabilities_[written_number.GetImageClass()] +=
        1.0 / written_numbers.size();
  }
}

} // namespace naivebayes
