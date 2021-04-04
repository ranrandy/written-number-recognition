#pragma once

#include <map>
#include <set>

#include "written_number.h"
#include "pixel.h"

namespace naivebayes {

/**
 * Process the data to get probabilities for P(class = c) and
 * P(F_{i, j} = f | class = c).
 */
class DataProcessor {
public:
  DataProcessor(const std::vector<WrittenNumber>& written_numbers);

  const std::map<std::string, double>& GetClassProbability() const;
  const std::map<Pixel, double>& GetPixelProbability() const;

private:
  void CalculateProbabilityForClasses(
      const std::vector<WrittenNumber>& written_numbers);

  void CalculateProbabilityForPixels(
      const std::vector<WrittenNumber>& written_numbers);

  std::map<std::string, double> image_class_probabilities_;
  std::map<Pixel, double> pixel_probabilities_;
};

} // namespace naivebayes

