#include "core/data_converter.h"

namespace naivebayes {

DataConverter::DataConverter(const std::string& file_path, size_t image_size) {
  file_path_ = file_path;
  image_size_ = image_size;
}

size_t DataConverter::GetImageClassCount() const {
  return image_classes_.size();
}

size_t DataConverter::GetGreatestWrittenNumber() const {
  return *max_element(image_classes_.begin(), image_classes_.end());
}

size_t DataConverter::GetImageSize() const {
  return image_size_;
}

const std::string& DataConverter::GetFilePath() const {
  return file_path_;
}

const std::vector<WrittenNumber>& DataConverter::GetDataset() const {
  return dataset_;
}

std::istream& operator>>(std::istream& in, DataConverter& data_converter) {
  std::ifstream data_file(data_converter.GetFilePath());

  int image_class;
  std::vector<std::vector<WrittenNumber::PixelColor>> image_vector;

  size_t line_count = 1;
  if (data_file.is_open()) {
    std::string line;
    while (getline(data_file, line)) {
      if (line_count % (data_converter.GetImageSize() + 1) == 1) {
        image_class = data_converter.ConvertToClass(line);
        data_converter.image_classes_.insert(image_class);
      } else {
        image_vector.push_back(data_converter.ConvertToPixels(line));
      }

      if (line_count % (data_converter.GetImageSize() + 1) == 0) {
        WrittenNumber written_number(image_class, image_vector);
        data_converter.dataset_.push_back(written_number);
        image_vector.clear();
      }
      line_count++;
      line.clear();
    }
  }
  data_file.close();
  return in;
}

int DataConverter::ConvertToClass(const std::string &line) {
  bool is_digit = true;
  for (char character : line) {
    is_digit = is_digit && std::isdigit(character);
  }

  if (is_digit) {
    return stoi(line);
  } else {
    return -1;
  }
}

std::vector<WrittenNumber::PixelColor> DataConverter::ConvertToPixels(
    const std::string& line) {
  std::vector<WrittenNumber::PixelColor> row_vector;
  for (char character : line) {
    if (character == kWhitePixel) {
      row_vector.push_back(WrittenNumber::PixelColor::kWhite);
    } else if (character == kGreyPixel) {
      row_vector.push_back(WrittenNumber::PixelColor::kGrey);
    } else if (character == kBlackPixel) {
      row_vector.push_back(WrittenNumber::PixelColor::kBlack);
    }
  }
  return row_vector;
}

} // namespace naivebayes

