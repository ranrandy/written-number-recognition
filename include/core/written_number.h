#pragma once

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

  /**
   * Constructor for this class, creates a new written number instance.
   * @param image_class the class of certain written number
   * @param image_vector the color info for a certain written number image
   */
  WrittenNumber(size_t image_class,
                const std::vector<std::vector<PixelColor>>& image_vector);

  /**
   * Gets the class of this written number.
   * @return the class of this written number
   */
  size_t GetImageClass() const;
  
  /**
   * Gets a 2D vector containing the color data for this written number image.
   * @return a 2D vector containing color for each pixel
   */
  const std::vector<std::vector<PixelColor>>& GetImageVector() const;

 private:
  size_t image_class_;
  std::vector<std::vector<PixelColor>> image_vector_;
};

} // namespace naivebayes







