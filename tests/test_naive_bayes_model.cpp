#include <catch2/catch.hpp>

#include "core/data_converter.h"
#include "core/naive_bayes_model.h"
#include "core/written_number.h"

using naivebayes::DataConverter;
using naivebayes::WrittenNumber;
using naivebayes::NaiveBayesModel;
using std::map;
using std::vector;
using std::string;

vector<string> SplitDataStrings(const string& line, char splitter); 

TEST_CASE("Calculate P(class = c)") {
  string data_file_path = "data/test_normal_images.txt";
  std::ifstream input_file(data_file_path);
  DataConverter data_converter;
  input_file >> data_converter;
  map<int, double> image_class_probabilities;

  SECTION("Default Processor") {
    NaiveBayesModel naive_bayes_model(data_converter);
    image_class_probabilities = naive_bayes_model.GetPriorProbability();
    REQUIRE(image_class_probabilities[4] == Approx(0.667).epsilon(0.01));
    REQUIRE(image_class_probabilities[6] == Approx(0.333).epsilon(0.01));
  }

  SECTION("After laplace smoothing") {
    NaiveBayesModel naive_bayes_model(data_converter, 10);
    image_class_probabilities = naive_bayes_model.GetPriorProbability();
    REQUIRE(image_class_probabilities[4] == Approx(0.5217).epsilon(0.01));
    REQUIRE(image_class_probabilities[6] == Approx(0.4782).epsilon(0.01));
  }
}

TEST_CASE("Calculate P(F_{i, j} = f | class = c)") {
  string data_file_path = "data/test_normal_images.txt";
  std::ifstream input_file(data_file_path);
  DataConverter data_converter;
  input_file >> data_converter;
  vector<vector<vector<vector<double>>>> pixel_probabilities;

  SECTION("Default Processor") {
    NaiveBayesModel naive_bayes_model(data_converter);
    pixel_probabilities = naive_bayes_model.GetConditionalProbability();
    REQUIRE(pixel_probabilities[14][16][2][4] == Approx(1).epsilon(0.01));
  }

  SECTION("After laplace smoothing") {
    NaiveBayesModel naive_bayes_model(data_converter, 10);
    pixel_probabilities = naive_bayes_model.GetConditionalProbability();
    REQUIRE(pixel_probabilities[14][16][2][4] == Approx(0.375).epsilon(0.01));
  }
}

TEST_CASE("Processing for arbitrary image sizes") {
  string data_file_path = "data/test_smaller_images.txt";
  std::ifstream input_file(data_file_path);
  DataConverter data_converter;
  input_file >> data_converter;
  map<int, double> image_class_probabilities;
  vector<vector<vector<vector<double>>>> pixel_probabilities;

  SECTION("Default Processor") {
    NaiveBayesModel naive_bayes_model(data_converter);
    image_class_probabilities = naive_bayes_model.GetPriorProbability();
    pixel_probabilities = naive_bayes_model.GetConditionalProbability();
    REQUIRE(image_class_probabilities[4] == Approx(0.667).epsilon(0.01));
    REQUIRE(pixel_probabilities[11][11][1][4] == Approx(1).epsilon(0.01));
  }

  SECTION("After laplace smoothing") {
    NaiveBayesModel naive_bayes_model(data_converter, 10);
    image_class_probabilities = naive_bayes_model.GetPriorProbability();
    pixel_probabilities = naive_bayes_model.GetConditionalProbability();
    REQUIRE(image_class_probabilities[1] == Approx(0.4782).epsilon(0.01));
    REQUIRE(pixel_probabilities[11][11][1][4] == Approx(0.375).epsilon(0.01));
  }
}

TEST_CASE("Writing trained model") {
  string data_file_path = "data/test_normal_images.txt";
  string output_file_path = "data/test_trained_model.txt";
  std::ifstream input_file(data_file_path);
  std::ofstream output_file(output_file_path);
  
  DataConverter data_converter;
  input_file >> data_converter;
  NaiveBayesModel naive_bayes_model(data_converter, 10);
  output_file << naive_bayes_model;

  std::ifstream trained_model_file(output_file_path);
  string line;
  getline(trained_model_file, line);
  vector<string> sub_strings = SplitDataStrings(line, ' ');
  REQUIRE(stoi(sub_strings.at(2)) == 7);
  
  for (size_t i = 0; i < 140; i++) {
    getline(trained_model_file, line);
  }
  sub_strings = SplitDataStrings(line, '\t');
  REQUIRE(stod(sub_strings.at(0)) == Approx(0.375).epsilon(0.01));
}

TEST_CASE("Loading from trained model") {
  NaiveBayesModel naive_bayes_model;
  std::ifstream input_file("data/computed_probabilities.txt");
  input_file >> naive_bayes_model;

  map<int, double> image_class_probabilities;
  vector<vector<vector<vector<double>>>> pixel_probabilities;
  image_class_probabilities = naive_bayes_model.GetPriorProbability();
  pixel_probabilities = naive_bayes_model.GetConditionalProbability();
 
  REQUIRE(image_class_probabilities[4] == Approx(0.1069).epsilon(0.001));
  REQUIRE(pixel_probabilities[11][11][1][4] == Approx(0.2182).epsilon(0.001));
}

vector<string> SplitDataStrings(const string& line, char splitter) {
  std::stringstream ss(line);
  string element;
  vector<string> sub_strings;
  while (getline(ss, element, splitter)) {
    sub_strings.push_back(element);
  }
  return sub_strings;
}
