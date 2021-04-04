#include "core/written_number.h"

namespace naivebayes {

WrittenNumber::WrittenNumber(
    char image_class,
    const std::vector<std::vector<PixelColor>>& image_vector) {
  image_class_ = image_class;
  image_vector_ = image_vector;
}

char WrittenNumber::GetImageClass() const {
  return image_class_;
}

const std::vector<std::vector<WrittenNumber::PixelColor>>&
WrittenNumber::GetImageVector() const {
  return image_vector_;
}

} // namespace naivebayes



