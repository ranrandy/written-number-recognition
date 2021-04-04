#include <catch2/catch.hpp>
#include <iostream>

#include "core/data_converter.h"
#include "core/data_processor.h"
#include "core/written_number.h"

using naivebayes::DataConverter;
using naivebayes::WrittenNumber;
using naivebayes::DataProcessor;

TEST_CASE("Calculate P(class = c)") {
  DataConverter data_converter(
      "/Users/lirunfeng/cinder_master/my-projects/naive-bayes-ranrandy/"
      "/data/data_for_test.txt", 28);
  std::cin >> data_converter;

  DataProcessor data_processor(data_converter.GetDataset());
  std::map<std::string, double> image_class_probabilities;
  auto it = data_processor.GetClassProbability().begin();
  while (it != data_processor.GetClassProbability().end()) {
    image_class_probabilities[it->first] = it->second;
    it++;
  }
  REQUIRE(image_class_probabilities["4"] == Approx(0.667).epsilon(0.01));
  REQUIRE(image_class_probabilities["6"] == Approx(0.333).epsilon(0.01));
}
