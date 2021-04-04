#include "core/data_processor.h"

namespace naivebayes {

DataProcessor::DataProcessor(
    const std::vector<WrittenNumber>& written_numbers) {
  CalculateProbabilityForClasses(written_numbers);
  CalculateProbabilityForPixels(written_numbers);
}

const std::map<std::string, double>&
    DataProcessor::GetClassProbability() const {
  return image_class_probabilities_;
}

const std::map<Pixel, double>& DataProcessor::GetPixelProbability() const {
  return pixel_probabilities_;
}

void DataProcessor::CalculateProbabilityForClasses(
    const std::vector<WrittenNumber>& written_numbers) {
  for (const WrittenNumber& written_number : written_numbers) {
    image_class_probabilities_[written_number.GetImageClass()] +=
        1.0 / written_numbers.size();
  }
}

void DataProcessor::CalculateProbabilityForPixels(
    const std::vector<WrittenNumber>& written_numbers) {
  for (const WrittenNumber& written_number : written_numbers) {
    for (size_t i = 0; i < written_number.GetImageVector().size(); i++) {
      for (size_t j = 0; j < written_number.GetImageVector()[i].size(); j++) {
        Pixel pixel(i, j, written_number.GetImageVector()[i][j],
                    image_class_probabilities_[written_number.GetImageClass()]);
        pixel_probabilities_[pixel] +=
            1.0 / (image_class_probabilities_[written_number.GetImageClass()] *
                   written_numbers.size());
      }
    }
  }
}

} // namespace naivebayes
