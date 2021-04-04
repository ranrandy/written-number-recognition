#include "core/pixel.h"

namespace naivebayes {

Pixel::Pixel(size_t row, size_t column,
             WrittenNumber::PixelColor color, double image_class_probability) {
  row_ = row;
  column_ = column;
  color_ = color;
  image_class_probability_ = image_class_probability;
}

} // namespace naivebayes