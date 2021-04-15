#include "core/dataset.h"

namespace naivebayes {

Dataset::Dataset() {
  image_size_ = 0;
}

size_t Dataset::GetImageClassCount() const {
  return image_classes_.size();
}

size_t Dataset::GetMaxWrittenNumber() const {
  return *max_element(image_classes_.begin(), image_classes_.end());
}

size_t Dataset::GetImageSize() const {
  return image_size_;
}

const std::vector<WrittenNumber>& Dataset::GetDataset() const {
  return dataset_;
}

std::istream& Dataset::operator<<(std::istream& data_file) {
  size_t image_class;
  std::vector<std::vector<WrittenNumber::PixelColor>> image_vector;

  std::string line;
  size_t line_count = 1;
  while (getline(data_file, line)) {
    if (ConvertToClass(line) >= 0) {
      // Then this line represents the class of the next image.
      image_class = ConvertToClass(line);
      image_classes_.insert(image_class);
    } else {
      // This way of getting image size only works when the image is a square.
      if (GetImageSize() == 0) {
        image_size_ = line.size();
      }
      image_vector.push_back(ConvertToPixels(line));
    }

    if (line_count % (GetImageSize() + 1) == 0 && GetImageSize() != 0) {
      WrittenNumber written_number(image_class, image_vector);
      dataset_.push_back(written_number);
      image_vector.clear();
    }
    line_count++;
  }
  return data_file;
}

int Dataset::ConvertToClass(const std::string &line) {
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

std::vector<WrittenNumber::PixelColor> Dataset::ConvertToPixels(
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

