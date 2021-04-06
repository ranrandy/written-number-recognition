#pragma once

#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>

#include "written_number.h"

namespace naivebayes {

/**
 * Convert the raw dataset so that it can be stored properly.
 */
class DataConverter {
 public:
  const size_t pixel_color_count_ = 3;

  DataConverter();

  size_t GetImageClassCount() const;
  size_t GetGreatestWrittenNumber() const;
  size_t GetImageSize() const;

  const std::vector<WrittenNumber>& GetDataset() const;

  friend std::istream &operator>>(std::istream& in,
                                  DataConverter& data_converter);

 private:
  const char kWhitePixel = ' ';
  const char kGreyPixel = '+';
  const char kBlackPixel = '#';

  int ConvertToClass(const std::string& line);
  std::vector<WrittenNumber::PixelColor> ConvertToPixels(
      const std::string& line);

  size_t image_size_;
  std::set<int> image_classes_;
  std::vector<WrittenNumber> dataset_;
};

} // namespace naivebayes

