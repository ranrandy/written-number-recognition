#include "core/data_processor.h"

namespace naivebayes {

DataProcessor::DataProcessor(
    const DataConverter& data_converter) {
  CalculateProbabilityForClasses(data_converter);

  // Initiate the pixel probabilities with 0.
  pixel_probabilities_ = vector<vector<vector<vector<double>>>>(
      data_converter.GetImageSize(),
      vector<vector<vector<double>>>(
          data_converter.GetImageSize(),
          vector<vector<double>>(
              data_converter.pixel_color_count_,
              vector<double>(
                  data_converter.greatest_written_number + 1, 0))));
  CalculateProbabilityForPixels(data_converter);
}

const std::map<int, double>& DataProcessor::GetClassProbability() const {
  return class_probabilities_;
}

vector<vector<vector<vector<double>>>> DataProcessor::GetPixelProbability()
    const {
  return pixel_probabilities_;
}

void DataProcessor::CalculateProbabilityForClasses(
    const DataConverter& data_converter) {
  std::vector<WrittenNumber> written_numbers = data_converter.GetDataset();

  for (const WrittenNumber& written_number : written_numbers) {
    if (written_number.GetImageClass() >= 0) {
      class_probabilities_[written_number.GetImageClass()]++;
    }
  }

  for (auto & it : class_probabilities_) {
     class_probabilities_[it.first] =
        (laplace_parameter + it.second) /
        (data_converter.image_class_count_ * laplace_parameter +
         written_numbers.size());
  }
}

void DataProcessor::CalculateProbabilityForPixels(
    const DataConverter& data_converter) {
  std::vector<WrittenNumber> written_numbers = data_converter.GetDataset();

  for (const WrittenNumber& written_number : written_numbers) {
    for (size_t i = 0; i < written_number.GetImageVector().size(); i++) {
      for (size_t j = 0; j < written_number.GetImageVector()[i].size(); j++) {
        pixel_probabilities_[i][j]
                            [static_cast<size_t>(
                                written_number.GetImageVector()[i][j])]
                            [written_number.GetImageClass()] =
            laplace_parameter /
            (class_probabilities_[written_number.GetImageClass()] *
                 written_numbers.size() +
             data_converter.pixel_color_count_ * laplace_parameter);
      }
    }
  }

  for (const WrittenNumber& written_number : written_numbers) {
    for (size_t i = 0; i < written_number.GetImageVector().size(); i++) {
      for (size_t j = 0; j < written_number.GetImageVector()[i].size(); j++) {
        pixel_probabilities_[i][j]
                            [static_cast<size_t>(
                                written_number.GetImageVector()[i][j])]
                            [written_number.GetImageClass()] +=
            1.0 / (class_probabilities_[written_number.GetImageClass()] *
                   written_numbers.size() +
                   data_converter.pixel_color_count_ * laplace_parameter);
      }
    }
  }
}

} // namespace naivebayes
