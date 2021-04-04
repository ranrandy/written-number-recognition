#pragma once

#include <map>
#include <set>

#include "core/data_converter.h"
#include "written_number.h"

namespace naivebayes {

using std::vector;

/**
 * Process the data to get probabilities for P(class = c) and
 * P(F_{i, j} = f | class = c).
 */
class DataProcessor {
public:
  DataProcessor(const DataConverter& data_converter);

  const std::map<size_t , double>& GetClassProbability() const;
  vector<vector<vector<vector<double>>>> GetPixelProbability() const;

private:
  void CalculateProbabilityForClasses(
      const DataConverter& data_converter);

  void CalculateProbabilityForPixels(
      const DataConverter& data_converter);

  std::map<size_t, double> image_class_probabilities_;
  vector<vector<vector<vector<double>>>> pixel_ps_vector;
};

} // namespace naivebayes

