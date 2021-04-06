#include "core/data_processor.h"

namespace naivebayes {

DataProcessor::DataProcessor() {}

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

std::ostream &operator<<(std::ostream& output_file, 
                         DataProcessor& data_processor) {
  // Outputs prior probabilities first.
  for (auto & it : data_processor.class_probabilities_) {
    output_file << it.first << " " << it.second << std::endl; 
  }
  
  // Inserts a line between prior probabilities and conditional probabilities.
  output_file << std::endl;
  
  size_t row_total = data_processor.pixel_probabilities_.size();
  size_t column_total = data_processor.pixel_probabilities_[0].size();
  size_t color_total = data_processor.pixel_probabilities_[0][0].size();
  size_t class_total = data_processor.pixel_probabilities_[0][0][0].size();
  
  // Outputs conditional probabilities.
  for (size_t class_count = 0; class_count < class_total; class_count++) {
    output_file << class_count << std::endl;
    for (size_t row = 0; row < row_total; row++) {
      for (size_t column = 0; column < column_total; column++) {
        for (size_t color_count = 0; color_count < color_total; color_count++) {
          output_file << std::fixed << std::setprecision(6) << 
              data_processor.pixel_probabilities_[row][column][color_count]
                                                 [class_count] << "\t";
        }
      }
      output_file << std::endl;
    }
    output_file << std::endl;
  }
  return output_file;
}

std::istream &operator>>(std::istream& input_file, 
                         DataProcessor& data_processor) {
  std::string line;
  
  // Reads prior probabilities
  while (getline(input_file, line)) {
    std::vector<std::string> sub_strings = 
        data_processor.SplitDataStrings(line, ' ');
    if (sub_strings.empty()) {
      break;
    } else {
      data_processor.class_probabilities_[stoi(sub_strings.at(0))] = 
          stod(sub_strings.at(1));
    }
  }
  
  // data_processor.InitiatePixelProbabilities();
  data_processor.pixel_probabilities_ = vector<vector<vector<vector<double>>>>(
      28, vector<vector<vector<double>>>(28, vector<vector<double>>(
                                                 3, vector<double>(10, 0))));
  // Reads conditional probabilities
  while (getline(input_file, line)) {
    std::vector<std::string> sub_strings =
        data_processor.SplitDataStrings(line, ' ');
    if (sub_strings.size() == 1) {
      size_t number_class = stoi(sub_strings.at(0));
      for (size_t i = 0; i < 28; i++) {
        getline(input_file, line);
        std::vector<std::string> col_data =
            data_processor.SplitDataStrings(line, '\t');
        for (size_t j = 0; j < 28; j++) {
          for (size_t color_count = 0; color_count < 3; color_count++) {
            data_processor.pixel_probabilities_[i][j][color_count]
                                               [number_class] = 
                stod(col_data.at(j * 3 + color_count));
          }
        }
      }
    }
  }
  return input_file;
}

std::vector<std::string> DataProcessor::SplitDataStrings(
    const std::string& line, char splitter) {
  std::stringstream ss(line);
  std::string element;
  std::vector<std::string> sub_strings;
  while (getline(ss, element, splitter)) {
    sub_strings.push_back(element);
  }
  return sub_strings;
}

} // namespace naivebayes
