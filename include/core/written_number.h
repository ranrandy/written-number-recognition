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

  WrittenNumber(const std::string& image_class,
                const std::vector<std::vector<PixelColor>>& image_vector);

  const std::string& GetImageClass() const;
  const std::vector<std::vector<PixelColor>>& GetImageVector() const;

 private:
  std::string image_class_;
  std::vector<std::vector<PixelColor>> image_vector_;
};

} // namespace naivebayes







