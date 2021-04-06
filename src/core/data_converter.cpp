#include "core/data_converter.h"

namespace naivebayes {

DataConverter::DataConverter() {
  image_size_ = 0;
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

const std::vector<WrittenNumber>& DataConverter::GetDataset() const {
  return dataset_;
}

std::istream& operator>>(std::istream& data_file, DataConverter& data_converter) {
  size_t image_class;
  std::vector<std::vector<WrittenNumber::PixelColor>> image_vector;

  std::string line;
  size_t line_count = 1;
  while (getline(data_file, line)) {
    if (data_converter.ConvertToClass(line) >= 0) {
      // Then this line represents the class of the next image.
      image_class = data_converter.ConvertToClass(line);
      data_converter.image_classes_.insert(image_class);
    } else {
      // This way of getting image size only works when the image is a square.
      if (data_converter.GetImageSize() == 0) {
        data_converter.image_size_ = line.size();
      }
      image_vector.push_back(data_converter.ConvertToPixels(line));
    }

    if (line_count % (data_converter.GetImageSize() + 1) == 0 &&
        data_converter.GetImageSize() != 0) {
      WrittenNumber written_number(image_class, image_vector);
      data_converter.dataset_.push_back(written_number);
      image_vector.clear();
    }
    line_count++;
  }
  return data_file;
}

int DataConverter::ConvertToClass(const std::string &line) {
  bool is_digit = true;
  if (line.empty()) {
    return -1;
  } else {
    for (char character : line) {
      is_digit = is_digit && std::isdigit(character);
    }
  }

  if (is_digit) {
    // Convert the line, which is a string, to int.
    return stoi(line);
  } else {
    return -1;
  }
}

std::vector<WrittenNumber::PixelColor> DataConverter::ConvertToPixels(
    const std::string& line) {
  std::vector<WrittenNumber::PixelColor> row_vector;
  for (char character : line) {
    if (character == kGreyPixel) {
      row_vector.push_back(WrittenNumber::PixelColor::kGrey);
    } else if (character == kBlackPixel) {
      row_vector.push_back(WrittenNumber::PixelColor::kBlack);
    } else {
      // If there exists some other characters other than whitespaces,
      // they will be converted to whitespaces in this case.
      row_vector.push_back(WrittenNumber::PixelColor::kWhite);
    }
  }
  return row_vector;
}

} // namespace naivebayes

