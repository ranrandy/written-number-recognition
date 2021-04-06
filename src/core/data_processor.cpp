#include "core/data_processor.h"

namespace naivebayes {

DataProcessor::DataProcessor(const DataConverter& data_converter,
                             double laplace_parameter) {
  laplace_parameter_ = laplace_parameter;
  CalculateProbabilityForClasses(data_converter);

  // Initiate the pixel probabilities with 0.
  pixel_probabilities_ = vector<vector<vector<vector<double>>>>(
      data_converter.GetImageSize(),
      vector<vector<vector<double>>>(
          data_converter.GetImageSize(),
          vector<vector<double>>(
              data_converter.pixel_color_count_,
              vector<double>(
                  data_converter.GetGreatestWrittenNumber() + 1, 0))));
  CalculateProbabilityForPixels(data_converter);
}

const std::map<int, double>& DataProcessor::GetClassProbability() const {
  return class_probabilities_;
}

const vector<vector<vector<vector<double>>>>&
DataProcessor::GetPixelProbability() const {
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
        (laplace_parameter_ + it.second) /
        (data_converter.GetImageClassCount() * laplace_parameter_ +
         written_numbers.size());
  }
}

void DataProcessor::CalculateProbabilityForPixels(
    const DataConverter& data_converter) {
  std::vector<WrittenNumber> written_numbers = data_converter.GetDataset();
  std::map<int, double> class_count;

  for (auto & it : class_probabilities_) {
    class_count[it.first] =
        it.second * (data_converter.GetDataset().size() +
                     data_converter.GetImageClassCount() *
                     laplace_parameter_) -
        laplace_parameter_;
  }

  for (const WrittenNumber& written_number : written_numbers) {
    for (size_t i = 0; i < written_number.GetImageVector().size(); i++) {
      for (size_t j = 0; j < written_number.GetImageVector()[i].size(); j++) {
        pixel_probabilities_[i][j]
                            [static_cast<size_t>(
                                written_number.GetImageVector()[i][j])]
                            [written_number.GetImageClass()] =
            laplace_parameter_ /
            (class_count[written_number.GetImageClass()] +
             data_converter.pixel_color_count_ * laplace_parameter_);
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
            1.0 / (class_count[written_number.GetImageClass()] +
                   data_converter.pixel_color_count_ * laplace_parameter_);
      }
    }
  }
}

} // namespace naivebayes
