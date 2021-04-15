#pragma once

#include <iostream>
#include <fstream>
#include <set>

#include "written_number.h"

namespace naivebayes {

/**
 * Convert the raw dataset so that it can be stored properly.
 */
class Dataset {
 public:
  // The number of pixel colors that will be in the raw dataset.
  // In this case, they are white, grey and black. 
  const size_t kPixelColorCount = 3;

  /**
   * Constructor of this class, only initialize image_size with 0.
   */
  Dataset();

  /**
   * Gets the number of written number classes in the dataset.
   * @return the number of written number classes
   */
  size_t GetImageClassCount() const;
  
  /**
   * Gets the largest number of written number in the dataset.
   * @return the largest value of written numbers
   */
  size_t GetMaxWrittenNumber() const;
  
  /**
   * Gets the image size of each written number in the dataset.
   * @return image size
   */
  size_t GetImageSize() const;

  /**
   * Gets a vector of all the written number instances gotten from the dataset.
   * @return a vector of all the written number instances.
   */
  const std::vector<WrittenNumber>& GetDataset() const;

  /**
   * Reads a dataset file.
   * @param data_file input file stream
   * @return file stream after reading the file
   */
  std::istream &operator<<(std::istream& data_file);

 private:
  const char kGreyPixel = '+';
  const char kBlackPixel = '#';

  /**
   * Converts a line in the dataset file to a number 
   * which is the class of the next written number image.
   * @param line a line in the dataset file
   * @return the class of the next written number image
   */
  int ConvertToClass(const std::string& line);
  
  /**
   * Converts a line in the dataset file to a vector of pixel colors
   * representing the colors of pixels in one row.
   * @param line a line in the dataset file
   * @return the colors of pixels in one row.
   */
  std::vector<WrittenNumber::PixelColor> ConvertToPixels(
      const std::string& line);

  size_t image_size_;
  
  // A set consisting of all possible written number classes in the dataset
  std::set<size_t> image_classes_;
  
  std::vector<WrittenNumber> dataset_;
};

} // namespace naivebayes

