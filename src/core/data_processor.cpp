#include "core/data_processor.h"

namespace naivebayes {

DataProcessor::DataProcessor(
    const DataConverter& data_converter) {
  CalculateProbabilityForClasses(data_converter);
  CalculateProbabilityForPixels(data_converter);
}

const std::map<size_t , double>&
    DataProcessor::GetClassProbability() const {
  return image_class_probabilities_;
}

vector<vector<vector<vector<double>>>> DataProcessor::GetPixelProbability()
    const {
  return pixel_ps_vector;
}

void DataProcessor::CalculateProbabilityForClasses(
    const DataConverter& data_converter) {
  std::vector<WrittenNumber> written_numbers = data_converter.GetDataset();
  for (const WrittenNumber& written_number : written_numbers) {
    image_class_probabilities_[written_number.GetImageClass()] +=
        1.0 / written_numbers.size();
  }
}

void DataProcessor::CalculateProbabilityForPixels(
    const DataConverter& data_converter) {
  std::vector<WrittenNumber> written_numbers = data_converter.GetDataset();

  for (size_t i = 0; i < 28; i++) {
    vector<vector<vector<double>>> pixel_ps_vector_3d;
    for (size_t j = 0; j < 28; j++) {
      vector<vector<double>> pixel_ps_vector_2d;
      for (size_t k = 0; k < 3; k++) {
        vector<double> pixel_ps_vector_1d;
        for (size_t l = 0; l < 10; l++) {
          pixel_ps_vector_1d.push_back(0);
        }
        pixel_ps_vector_2d.push_back(pixel_ps_vector_1d);
      }
      pixel_ps_vector_3d.push_back(pixel_ps_vector_2d);
    }
    pixel_ps_vector.push_back(pixel_ps_vector_3d);
  }

  for (const WrittenNumber& written_number : written_numbers) {
    for (size_t i = 0; i < written_number.GetImageVector().size(); i++) {
      for (size_t j = 0; j < written_number.GetImageVector()[i].size(); j++) {
        pixel_ps_vector[i][j]
                       [static_cast<size_t>(
                           written_number.GetImageVector()[i][j])]
                       [written_number.GetImageClass()] +=
            1.0 / (image_class_probabilities_[written_number.GetImageClass()] *
                   written_numbers.size());
      }
    }
  }

}

} // namespace naivebayes
