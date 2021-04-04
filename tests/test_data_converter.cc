#include <catch2/catch.hpp>
#include <iostream>

#include "core/data_converter.h"
#include "core/written_number.h"

using naivebayes::DataConverter;
using naivebayes::WrittenNumber;

TEST_CASE("Overload >> operator to read the data") {
  DataConverter data_converter(
      "/Users/lirunfeng/cinder_master/my-projects/naive-bayes-ranrandy/"
      "/data/data_for_test.txt", 28);
  std::cin >> data_converter;

  REQUIRE(data_converter.GetDataset()[2].GetImageClass() == 4);
  REQUIRE(data_converter.GetDataset()[1].GetImageVector().at(2).at(16) ==
          WrittenNumber::PixelColor::kBlack);
}
