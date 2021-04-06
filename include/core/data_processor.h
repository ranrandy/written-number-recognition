#pragma once

#include <map>

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

  /**
   * Gets a map containing P(class = c) data.
   * @return a map(class = c, P(class = c))
   */
  const std::map<int, double>& GetClassProbability() const;

  /**
   * Gets a 4D vector containing P(F_{i, j} = f | class = c) data.
   * @return a vector consisting of P(F_{i, j} = f | class = c)
   */
  vector<vector<vector<vector<double>>>> GetPixelProbability() const;

private:
  const double laplace_parameter = 10.5;

  void CalculateProbabilityForClasses(const DataConverter& data_converter);
  void CalculateProbabilityForPixels(const DataConverter& data_converter);

  // P(class = c)
  std::map<int, double> class_probabilities_;

  // P(F_{i, j} = f | class = c)
  vector<vector<vector<vector<double>>>> pixel_probabilities_;
};

} // namespace naivebayes

