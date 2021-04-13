#include <catch2/catch.hpp>

#include "core/k_nearest_neighbors_classifier.h"

using naivebayes::KNearestNeighborClassifier;
using naivebayes::DataConverter;

TEST_CASE("Classify one written number") {
  DataConverter dataset_converter;
  DataConverter test_data_converter;
  
  std::ifstream dataset_file("data/test_normal_images.txt");
  std::ifstream test_data_file("data/test_one_written_number.txt");
  
  dataset_converter << dataset_file;
  test_data_converter << test_data_file;
  
  KNearestNeighborClassifier knn;
  size_t result = knn.Classify(test_data_converter.GetDataset().front(), 
                               dataset_converter, 2);
  REQUIRE(result == 4);
}