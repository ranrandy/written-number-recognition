#include "core/k_nearest_neighbors_classifier.h"

namespace naivebayes {

KNearestNeighborClassifier::KNearestNeighborClassifier() {}

double KNearestNeighborClassifier::EvaluateAccuracy(
    const Dataset& test_data_converter, 
    const Dataset& dataset_converter, 
    size_t k) {
  size_t correct_result_count = 0;
  for (const WrittenNumber& written_number : 
       test_data_converter.GetDataset()) {
    size_t classified_result = Classify(written_number, dataset_converter, k);
    
    testing_results_.push_back(written_number.GetImageClass());
    classification_results_.push_back(classified_result);
    
    // Stores if this number is correctly classified.
    if (classified_result == written_number.GetImageClass()) {
      correct_result_count++;
    }
  }
  return double(correct_result_count) / 
         double(test_data_converter.GetDataset().size());
}

size_t KNearestNeighborClassifier::Classify(
    const WrittenNumber& written_number, 
    const Dataset& dataset_converter, 
    size_t k) {
  // Stores the distance between each image in the dataset and the image 
  // waiting to be classified.
  std::vector<std::pair<double, size_t>> distances;

  // Calculate the distances and add them to the distances vector.
  for (const WrittenNumber& classified_number :
      dataset_converter.GetDataset()) {
    std::pair<double, size_t> distance =
        CalculateDistance(written_number, classified_number);
    distances.push_back(distance);
  }

  // Sorts the distances in ascending order.
  sort(distances.begin(), distances.end());

  // Calculate the occurrences of each written number 
  // in the k smallest distances.
  std::map<size_t, size_t> class_occurrences;
  for (size_t i = 0; i < k; i++) {
    class_occurrences[distances[i].second] += 1;
  }

  // Gets the written number that occurs the largest number of times 
  // in the k nearest neighbors.
  size_t classified_result;
  size_t occurrence = 0;
  for (auto & class_count : class_occurrences) {
    if (class_count.second > occurrence) {
      classified_result = class_count.first;
      occurrence = class_count.second;
    }
  }
  return classified_result;
}

const std::vector<size_t>& KNearestNeighborClassifier::GetTestingResults() 
    const {
  return testing_results_;
}

const std::vector<size_t>& KNearestNeighborClassifier::
    GetClassificationResults() const {
  return classification_results_;
}

void KNearestNeighborClassifier::OutputConfusingMatrix() {
  std::vector<size_t> predicted_row(class_total, 0);
  std::vector<std::vector<size_t>> matrix(class_total, predicted_row);

  for (size_t actual = 0; actual < class_total; actual++) {
    for (size_t predicted = 0; predicted < class_total; predicted++) {
      size_t total = 0;
      for (size_t i = 0; i < GetTestingResults().size(); i++) {
        if (GetTestingResults().at(i) == actual &&
            GetClassificationResults().at(i) == predicted) {
          total++;
        }
      }
      matrix[actual][predicted] = total;
      std::cout << total << "\t";
    }
    std::cout << std::endl;
  }
}

std::pair<double, size_t> KNearestNeighborClassifier::CalculateDistance(
    const WrittenNumber& test_number, const WrittenNumber& data_number) {
  double square_distance_sum = 0;
  for (size_t i = 0; i < test_number.GetImageVector().size(); i++) {
    for (size_t j = 0; j < test_number.GetImageVector()[i].size(); j++) {
      square_distance_sum +=
          pow(static_cast<int>(test_number.GetImageVector()[i][j]) -
              static_cast<int>(
                  data_number.GetImageVector()[i][j]), 2);
    }
  }
  std::pair<double, size_t> distance(square_distance_sum,
                                     data_number.GetImageClass());
  return distance;
}

} // namespace naivebayes