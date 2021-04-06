#include "core/data_processor.h"

namespace naivebayes {

DataProcessor::DataProcessor(const DataConverter& data_converter,
                             double laplace_parameter) {
  laplace_parameter_ = laplace_parameter;
  CountClasses(data_converter);
  CalculateProbabilityForClasses(data_converter);
  InitiatePixelProbabilities(data_converter);
  CalculateProbabilityForPixels(data_converter);
}

const std::map<int, double>& DataProcessor::GetClassProbability() const {
  return class_probabilities_;
}

const vector<vector<vector<vector<double>>>>&
DataProcessor::GetPixelProbability() const {
  return pixel_probabilities_;
}

void DataProcessor::CountClasses(const DataConverter& data_converter) {
  for (const WrittenNumber& written_number : data_converter.GetDataset()) {
    if (written_number.GetImageClass() >= 0) {
      class_count_[written_number.GetImageClass()]++;
    }
  }
}

void DataProcessor::CalculateProbabilityForClasses(
    const DataConverter& data_converter) {
  for (auto & it : class_count_) {
     class_probabilities_[it.first] =
        (laplace_parameter_ + it.second) /
        (data_converter.GetImageClassCount() * laplace_parameter_ +
         data_converter.GetDataset().size());
  }
}

void DataProcessor::InitiatePixelProbabilities(
    const DataConverter& data_converter) {
  // Initializes a 4D vector whose size info for each layer is 
  // (image_size * image_size * pixel_color_count * greatest_written_number)
  pixel_probabilities_ = vector<vector<vector<vector<double>>>>(
      data_converter.GetImageSize(),
      vector<vector<vector<double>>>(
          data_converter.GetImageSize(),
          vector<vector<double>>(
              data_converter.kPixelColorCount,
              vector<double>(
                  data_converter.GetGreatestWrittenNumber() + 1, 0))));

  // Assigns each P(F_{i, j} = f | class = c) with
  // k / (pixel_color_count * k + # classes belonging to class c).
  for (const WrittenNumber& written_number : data_converter.GetDataset()) {
    for (size_t i = 0; i < written_number.GetImageVector().size(); i++) {
      for (size_t j = 0; j < written_number.GetImageVector()[i].size(); j++) {
        pixel_probabilities_[i][j]
        [static_cast<size_t>(
            written_number.GetImageVector()[i][j])]
        [written_number.GetImageClass()] =
            laplace_parameter_ /
            (class_count_[written_number.GetImageClass()] +
             data_converter.kPixelColorCount * laplace_parameter_);
      }
    }
  }
}

void DataProcessor::CalculateProbabilityForPixels(
    const DataConverter& data_converter) {
  for (const WrittenNumber& written_number : data_converter.GetDataset()) {
    for (size_t i = 0; i < written_number.GetImageVector().size(); i++) {
      for (size_t j = 0; j < written_number.GetImageVector()[i].size(); j++) {
        pixel_probabilities_[i][j]
                            [static_cast<size_t>(
                                written_number.GetImageVector()[i][j])]
                            [written_number.GetImageClass()] +=
            1.0 / (class_count_[written_number.GetImageClass()] +
                   data_converter.kPixelColorCount * laplace_parameter_);
      }
    }
  }
}

} // namespace naivebayes
