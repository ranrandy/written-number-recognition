#include "core/data_converter.h"

namespace naivebayes {

DataConverter::DataConverter(const std::string& file_path, size_t image_size) {
  file_path_ = file_path;
  image_size_ = image_size;
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

  std::vector<std::vector<WrittenNumber::PixelColor>> image_vector;
  std::string image_class;
  std::string image_str;

  size_t line_count = 1;
  if (data_file.is_open()) {
    std::string line;
    while (getline(data_file, line)) {
      if (line_count % (data_converter.GetImageSize() + 1) == 1) {
        image_class = line;
      } else {
        std::vector<WrittenNumber::PixelColor> row_vector;
        for (char character : line) {
          if (character == data_converter.kWhitePixel) {
            row_vector.push_back(WrittenNumber::PixelColor::kWhite);
          } else if (character == data_converter.kGreyPixel) {
            row_vector.push_back(WrittenNumber::PixelColor::kGrey);
          } else if (character == data_converter.kBlackPixel) {
            row_vector.push_back(WrittenNumber::PixelColor::kBlack);
          }
        }
        image_vector.push_back(row_vector);
      }

      if (line_count % (data_converter.GetImageSize() + 1) == 0) {
        WrittenNumber written_number(image_class, image_vector);
        data_converter.dataset_.push_back(written_number);
        image_vector.clear();
        image_str.clear();
      }
      line_count++;
      image_str += line;
      line.clear();
    }
  }
  data_file.close();
  return in;
}

} // namespace naivebayes

