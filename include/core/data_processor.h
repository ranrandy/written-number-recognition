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
  DataProcessor(const DataConverter& data_converter,
                double laplace_parameter = 0);

  /**
   * Gets a map containing P(class = c) data.
   * @return a map(class = c, P(class = c))
   */
  const std::map<int, double>& GetClassProbability() const;

  /**
   * Gets a 4D vector containing P(F_{i, j} = f | class = c) data.
   * @return a vector consisting of P(F_{i, j} = f | class = c)
   */
  const vector<vector<vector<vector<double>>>>& GetPixelProbability() const;

private:
  void CountClasses(const DataConverter& data_converter);
  void CalculateProbabilityForClasses(const DataConverter& data_converter);
  void InitiatePixelProbabilities(const DataConverter& data_converter);
  void CalculateProbabilityForPixels(const DataConverter& data_converter);

  // Parameter k used for laplace smoothing
  double laplace_parameter_;

  // The number of each class in the dataset
  std::map<int, size_t> class_count_;

  // P(class = c)
  std::map<int, double> class_probabilities_;

  // P(F_{i, j} = f | class = c)
  vector<vector<vector<vector<double>>>> pixel_probabilities_;
};

} // namespace naivebayes

