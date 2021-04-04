#pragma once

#include <iostream>
#include <fstream>

#include "written_number.h"

namespace naivebayes {

class DataConverter {
 public:
  DataConverter(const std::string& file_path, size_t image_size);

  size_t GetImageSize() const;

  const std::string& GetFilePath() const;

  const std::vector<WrittenNumber>& GetDataset() const;

  friend std::istream &operator>>(std::istream& in,
                                  DataConverter& data_converter);

 private:
  const char kWhitePixel = ' ';
  const char kGreyPixel = '+';
  const char kBlackPixel = '#';

  size_t image_size_;
  std::string file_path_;
  std::vector<WrittenNumber> dataset_;
};

} // namespace naivebayes

