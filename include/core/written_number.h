#pragma once

#include <string>
#include <vector>

namespace naivebayes {

/**
 * Represents each written number in a dataset.
 */
class WrittenNumber {
 public:
  enum class PixelColor {
     kWhite,
     kGrey,
     kBlack
  };

  WrittenNumber(size_t image_class,
                const std::vector<std::vector<PixelColor>>& image_vector);

  size_t GetImageClass() const;
  const std::vector<std::vector<PixelColor>>& GetImageVector() const;

 private:
  size_t image_class_;
  std::vector<std::vector<PixelColor>> image_vector_;
};

} // namespace naivebayes







