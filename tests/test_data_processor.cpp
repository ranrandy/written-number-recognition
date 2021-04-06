#include <catch2/catch.hpp>
#include <iostream>

#include "core/data_converter.h"
#include "core/data_processor.h"
#include "core/written_number.h"

using naivebayes::DataConverter;
using naivebayes::WrittenNumber;
using naivebayes::DataProcessor;
using std::cin;
using std::map;
using std::vector;

TEST_CASE("Calculate P(class = c)") {
  std::string data_file_path = "/Users/lirunfeng/cinder_master/my-projects/"
                               "naive-bayes-ranrandy/data/test_normal_images"
                               ".txt";
  std::ifstream input_file(data_file_path);
  DataConverter data_converter;
  input_file >> data_converter;
  map<int, double> image_class_probabilities;

  SECTION("Default Processor") {
    DataProcessor data_processor(data_converter);
    image_class_probabilities = data_processor.GetClassProbability();
    REQUIRE(image_class_probabilities[4] == Approx(0.667).epsilon(0.01));
    REQUIRE(image_class_probabilities[6] == Approx(0.333).epsilon(0.01));
  }

  SECTION("After laplace smoothing") {
    DataProcessor data_processor(data_converter, 10);
    image_class_probabilities = data_processor.GetClassProbability();
    REQUIRE(image_class_probabilities[4] == Approx(0.5217).epsilon(0.01));
    REQUIRE(image_class_probabilities[6] == Approx(0.4782).epsilon(0.01));
  }
}

TEST_CASE("Calculate P(F_{i, j} = f | class = c)") {
  std::string data_file_path = "/Users/lirunfeng/cinder_master/my-projects/"
                               "naive-bayes-ranrandy/data/test_normal_images"
                               ".txt";
  std::ifstream input_file(data_file_path);
  DataConverter data_converter;
  input_file >> data_converter;
  vector<vector<vector<vector<double>>>> pixel_probabilities;

  SECTION("Default Processor") {
    DataProcessor data_processor(data_converter);
    pixel_probabilities = data_processor.GetPixelProbability();
    REQUIRE(pixel_probabilities[14][16][2][4] == Approx(1).epsilon(0.01));
  }

  SECTION("After laplace smoothing") {
    DataProcessor data_processor(data_converter, 10);
    pixel_probabilities = data_processor.GetPixelProbability();
    REQUIRE(pixel_probabilities[14][16][2][4] == Approx(0.375).epsilon(0.01));
  }
}

TEST_CASE("Processing for arbitrary image sizes") {
  std::string data_file_path = "/Users/lirunfeng/cinder_master/my-projects/"
                               "naive-bayes-ranrandy/data/test_smaller_images"
                               ".txt";
  std::ifstream input_file(data_file_path);
  DataConverter data_converter;
  input_file >> data_converter;
  map<int, double> image_class_probabilities;
  vector<vector<vector<vector<double>>>> pixel_probabilities;

  SECTION("Default Processor") {
    DataProcessor data_processor(data_converter);
    image_class_probabilities = data_processor.GetClassProbability();
    pixel_probabilities = data_processor.GetPixelProbability();
    REQUIRE(image_class_probabilities[4] == Approx(0.667).epsilon(0.01));
    REQUIRE(pixel_probabilities[11][11][1][4] == Approx(1).epsilon(0.01));
  }

  SECTION("After laplace smoothing") {
    DataProcessor data_processor(data_converter, 10);
    image_class_probabilities = data_processor.GetClassProbability();
    pixel_probabilities = data_processor.GetPixelProbability();
    REQUIRE(image_class_probabilities[1] == Approx(0.4782).epsilon(0.01));
    REQUIRE(pixel_probabilities[11][11][1][4] == Approx(0.375).epsilon(0.01));
  }
}

TEST_CASE("Loading from trained model") {
  DataProcessor data_processor;
  std::ifstream input_file("/Users/lirunfeng/cinder_master/my-projects/"
                           "naive-bayes-ranrandy/data/computed_probabilities"
                           ".txt");
  input_file >> data_processor;

  map<int, double> image_class_probabilities;
  vector<vector<vector<vector<double>>>> pixel_probabilities;
  image_class_probabilities = data_processor.GetClassProbability();
  pixel_probabilities = data_processor.GetPixelProbability();
 
  REQUIRE(image_class_probabilities[4] == Approx(0.1058).epsilon(0.001));
  REQUIRE(pixel_probabilities[11][11][1][4] == Approx(0.2419).epsilon(0.001));
}

