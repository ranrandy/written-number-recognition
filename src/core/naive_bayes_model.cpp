#include "core/naive_bayes_model.h"

namespace naivebayes {

NaiveBayesModel::NaiveBayesModel() {}

NaiveBayesModel::NaiveBayesModel(const DataConverter& data_converter,
                             double laplace_parameter) {
  laplace_parameter_ = laplace_parameter;
  CountClasses(data_converter);
  CalculateProbabilityForClasses(data_converter);
  InitiatePixelProbabilities(data_converter.GetImageSize(), 
                             data_converter.kPixelColorCount, 
                             data_converter.GetMaxWrittenNumber());
  CalculateProbabilityForPixels(data_converter);
}

const std::map<int, double>& NaiveBayesModel::GetPriorProbability() const {
  return class_probabilities_;
}

const NaiveBayesModel::vec4& NaiveBayesModel::GetConditionalProbability() 
    const {
  return pixel_probabilities_;
}

void NaiveBayesModel::CountClasses(const DataConverter& data_converter) {
  for (const WrittenNumber& written_number : data_converter.GetDataset()) {
    if (written_number.GetImageClass() >= 0) {
      class_count_[written_number.GetImageClass()]++;
    }
  }
}

void NaiveBayesModel::CalculateProbabilityForClasses(
    const DataConverter& data_converter) {
  for (auto & it : class_count_) {
     class_probabilities_[it.first] =
        (laplace_parameter_ + it.second) /
        (data_converter.GetImageClassCount() * laplace_parameter_ +
         data_converter.GetDataset().size());
  }
}

void NaiveBayesModel::InitiatePixelProbabilities(size_t image_size, 
                                               size_t color_count, 
                                               size_t max_number) {
  // Initializes a 4D vector whose size info for each layer is 
  // (image_size * image_size * pixel_color_count * greatest_written_number)
  pixel_probabilities_ = vec4(image_size, 
                              vec3(image_size, 
                                   vec2(color_count, 
                                        vector<double>(max_number + 1, 0))));
}

void NaiveBayesModel::CalculateProbabilityForPixels(
    const DataConverter& data_converter) {
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

std::ostream& NaiveBayesModel::operator>>(std::ostream& output_file) {
  size_t image_size = pixel_probabilities_.size();
  size_t color_total = pixel_probabilities_[0][0].size();
  size_t max_number = pixel_probabilities_[0][0][0].size();
  
  // Outputs image size, number of colors, number of written number classes
  // and the largest written number.
  output_file << image_size << " " << color_total << " " << max_number << 
      " " << std::endl;
  
  // Outputs prior probabilities first.
  for (auto & it : class_probabilities_) {
    output_file << it.first << " " << it.second << std::endl; 
  }
  
  // Outputs a blank line between prior and conditional probabilities.
  output_file << std::endl;
  
  // Outputs conditional probabilities.
  for (size_t class_count = 0; class_count < max_number; class_count++) {
    output_file << class_count << std::endl;
    for (size_t row = 0; row < image_size; row++) {
      for (size_t column = 0; column < image_size; column++) {
        for (size_t color_count = 0; color_count < color_total; color_count++) {
          output_file << std::fixed << std::setprecision(6) << 
              pixel_probabilities_[row][column][color_count]
                                  [class_count] << "\t";
        }
      }
      output_file << std::endl;
    }
  }
  return output_file;
}

std::istream& NaiveBayesModel::operator<<(std::istream& input_file) {
  std::string line;
  
  // Reads the image size, number of colors and number of written 
  // number classes from the first line of the file.
  getline(input_file, line);
  vector<std::string> data_parameters = SplitDataStrings(line, ' ');
  size_t image_size = stoi(data_parameters.at(0));
  size_t color_total = stoi(data_parameters.at(1));
  size_t max_number = stoi(data_parameters.at(2));
  
  // Reads prior probabilities
  while (getline(input_file, line)) {
    vector<std::string> class_pair = SplitDataStrings(line, ' ');
    if (class_pair.empty()) {
      break;
    } else {
      class_probabilities_[stoi(class_pair.at(0))] = stod(class_pair.at(1));
    }
  }
  
  // Reads conditional probabilities
  InitiatePixelProbabilities(image_size, color_total, 
                                            max_number);
  while (getline(input_file, line)) {
    vector<std::string> sub_strings = SplitDataStrings(line, ' ');
    if (sub_strings.size() == 1) {
      // Reads probabilities for a certain number class
      size_t number_class = stoi(sub_strings.at(0));
      for (size_t i = 0; i < image_size; i++) {
        getline(input_file, line);
        vector<std::string> col_data = SplitDataStrings(line, '\t');
        
        // Reads probabilities from one row for a certain number class
        for (size_t j = 0; j < image_size; j++) {
          for (size_t color_count = 0; color_count < color_total; 
               color_count++) {
            pixel_probabilities_[i][j][color_count][number_class] = 
                stod(col_data.at(j * color_total + color_count));
          }
        }
      }
    }
  }
  return input_file;
}

vector<std::string> NaiveBayesModel::SplitDataStrings(
    const std::string& line, char splitter) {
  std::stringstream ss(line);
  std::string element;
  vector<std::string> sub_strings;
  while (getline(ss, element, splitter)) {
    sub_strings.push_back(element);
  }
  return sub_strings;
}

} // namespace naivebayes
