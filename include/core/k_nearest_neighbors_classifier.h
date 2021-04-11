#pragma once

#include <map>

#include "core/data_converter.h"
#include "core/written_number.h"

namespace naivebayes {

/**
 * Classify handwritten numbers using k-nearest neighbors algorithm.
 */
class KNearestNeighborClassifier {
public:
  KNearestNeighborClassifier();

  double EvaluateAccuracy(const DataConverter& test_data_converter, 
                          const DataConverter& dataset_converter, 
                          double k);
};

} // namespace knn

