#pragma once

#include <iostream>

#include "core/written_number.h"

namespace naivebayes {

/**
 * Represents a pixel in an image.
 */
class Pixel {
 public:
  Pixel(size_t row, size_t column, WrittenNumber::PixelColor color,
        double image_class_probability);

 private:
  size_t row_;
  size_t column_;
  WrittenNumber::PixelColor color_;
  double image_class_probability_;
};

} // namespace naivebayes

