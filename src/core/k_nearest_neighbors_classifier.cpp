#include "core/k_nearest_neighbors_classifier.h"

namespace naivebayes {

KNearestNeighborClassifier::KNearestNeighborClassifier() {}

double KNearestNeighborClassifier::EvaluateAccuracy(
    const DataConverter& test_data_converter, 
    const DataConverter& dataset_converter, 
    double k) {
  size_t correct_result_count = 0;
  for (const WrittenNumber& written_number : 
       test_data_converter.GetDataset()) {
    std::vector<std::pair<double, size_t>> distances;
    for (const WrittenNumber& classified_number : 
         dataset_converter.GetDataset()) {
      double square_distance_sum = 0;
      for (size_t i = 0; i < dataset_converter.GetImageSize(); i++) {
        for (size_t j = 0; j < dataset_converter.GetImageSize(); j++) {
          square_distance_sum += 
              pow(static_cast<int>(written_number.GetImageVector()[i][j]) - 
                      static_cast<int>(
                          classified_number.GetImageVector()[i][j]), 2);
        }
      }
      std::pair<double, size_t> distance(square_distance_sum, 
                                         classified_number.GetImageClass());
      distances.push_back(distance);
    }
    
    sort(distances.begin(), distances.end());
    
    std::map<size_t, size_t> class_occurrences;
    for (size_t i = 0; i < k; i++) {
      class_occurrences[distances[i].second] += 1;
    }
    
    size_t classified_result;
    size_t occurrence = 0;
    for (auto & it : class_occurrences) {
      if (it.second > occurrence) {
        classified_result = it.first;
        occurrence = it.second;
      }
    }
    
    if (classified_result == written_number.GetImageClass()) {
      correct_result_count++;
    }
  }
  return double(correct_result_count) / 
         double(test_data_converter.GetDataset().size());
}


} // namespace naivebayes