#pragma once

#include <map>

#include "core/dataset.h"
#include "core/written_number.h"

namespace naivebayes {

/**
 * Classify handwritten numbers using k-nearest neighbors algorithm.
 */
class KNearestNeighborClassifier {
 public:
  /**
   * Default Constructor. Constructs a new k nearest neighbor classifier.
   */
  KNearestNeighborClassifier();

  /**
   * Evaluates the accuracy of using one dataset to classify another
   * testing dataset.
   * @param test_data_converter the dataset to be classified
   * @param dataset_converter the dataset used to classify
   * @param k the parameter k in k nearest neighbors algorithm
   * @return the accuracy of the algorithm
   */
  double EvaluateAccuracy(const DataConverter& test_data_converter, 
                          const DataConverter& dataset_converter, 
                          size_t k);
  
  /**
   * Classify one written number.
   * @param written_number the image to be classified
   * @param dataset_converter the dataset used to classify
   * @param k the parameter k in k nearest neighbors algorithm
   * @return the classification result of the image
   */
  size_t Classify(const WrittenNumber& written_number, 
                  const DataConverter& dataset_converter, 
                  size_t k);
  
 private:   
  std::pair<double, size_t> CalculateDistance(
       const WrittenNumber& test_number, const WrittenNumber& data_number);
};

} // namespace knn

