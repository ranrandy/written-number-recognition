#include <catch2/catch.hpp>

#include "core/dataset.h"
#include "core/written_number.h"

using naivebayes::DataConverter;
using naivebayes::WrittenNumber;

TEST_CASE("Overload << operator to read the data") {
  DataConverter data_converter;
  
  SECTION("Empty dataset") {
    std::ifstream input_file("data/test_empty_dataset.txt");
    data_converter << input_file;
    REQUIRE(data_converter.GetImageSize() == 0);
    REQUIRE(data_converter.GetDataset().empty());
    REQUIRE(data_converter.GetImageClassCount() == 0);
    REQUIRE(data_converter.GetMaxWrittenNumber() == 0);
  }
  
  SECTION("Normal image size data (size = 28)") {
    std::ifstream input_file("data/test_normal_images.txt");
    data_converter << input_file;
    REQUIRE(data_converter.GetDataset()[2].GetImageClass() == 4);
    REQUIRE(data_converter.GetDataset()[1].GetImageVector().at(3).at(18) ==
            WrittenNumber::PixelColor::kBlack);
  }
  
  SECTION("Arbitrary image size data (size = 20)") {
    std::ifstream input_file("data/test_smaller_images.txt");
    data_converter << input_file;
    REQUIRE(data_converter.GetDataset()[1].GetImageClass() == 1);
    REQUIRE(data_converter.GetDataset()[1].GetImageVector().at(10).at(13) ==
            WrittenNumber::PixelColor::kBlack);
  }
}
