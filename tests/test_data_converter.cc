#include <catch2/catch.hpp>
#include <iostream>

#include "core/data_converter.h"
#include "core/written_number.h"

using naivebayes::DataConverter;
using naivebayes::WrittenNumber;

TEST_CASE("Overload >> operator to read the data") {
  DataConverter dataConverter(
      "/Users/lirunfeng/cinder_master/my-projects/naive-bayes-ranrandy/"
      "/data/data_for_test.txt", 28);
  std::cin >> dataConverter;

  REQUIRE(dataConverter.GetDataset()[2].GetImageClass() == '4');
  REQUIRE(dataConverter.GetDataset()[1].GetImageVector().at(17).at(17) ==
          WrittenNumber::PixelColor::kBlack);
}
