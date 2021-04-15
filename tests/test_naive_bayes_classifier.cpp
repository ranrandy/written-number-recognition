#include <catch2/catch.hpp>

#include "core/dataset.h"
#include "core/naive_bayes_classifier.h"
#include "core/written_number.h"

using naivebayes::DataConverter;
using naivebayes::WrittenNumber;
using naivebayes::NaiveBayesClassifier;
using std::map;
using std::vector;
using std::string;

vector<string> SplitDataStrings(const string& line, char splitter); 

TEST_CASE("Calculate P(class = c), the prior probability") {
  string data_file_path = "data/test_normal_images.txt";
  std::ifstream input_file(data_file_path);
  DataConverter data_converter;
  data_converter << input_file;
  map<int, double> image_class_probabilities;

  SECTION("Default Processor") {
    NaiveBayesClassifier naive_bayes_model;
    naive_bayes_model.Train(data_converter);
    image_class_probabilities = naive_bayes_model.GetPriorProbability();
    REQUIRE(image_class_probabilities[4] == Approx(0.667).epsilon(0.01));
    REQUIRE(image_class_probabilities[6] == Approx(0.333).epsilon(0.01));
  }

  SECTION("After laplace smoothing") {
    NaiveBayesClassifier naive_bayes_model;
    naive_bayes_model.Train(data_converter, 10);
    image_class_probabilities = naive_bayes_model.GetPriorProbability();
    REQUIRE(image_class_probabilities[4] == Approx(0.5217).epsilon(0.01));
    REQUIRE(image_class_probabilities[6] == Approx(0.4782).epsilon(0.01));
  }
}

TEST_CASE("Calculate P(F_{i, j} = f | class = c), "
          "the conditional probability") {
  string data_file_path = "data/test_normal_images.txt";
  std::ifstream input_file(data_file_path);
  DataConverter data_converter;
  data_converter << input_file;
  vector<vector<vector<vector<double>>>> pixel_probabilities;

  SECTION("Default Processor") {
    NaiveBayesClassifier naive_bayes_model;
    naive_bayes_model.Train(data_converter);
    pixel_probabilities = naive_bayes_model.GetConditionalProbability();
    REQUIRE(pixel_probabilities[14][16][2][4] == Approx(1).epsilon(0.01));
    REQUIRE(pixel_probabilities[12][10][0][4] == Approx(0.5).epsilon(0.01));
    REQUIRE(pixel_probabilities[11][9][1][4] == Approx(0).epsilon(0.01));
    REQUIRE(pixel_probabilities[8][16][2][4] == Approx(0).epsilon(0.01));
  }

  SECTION("After laplace smoothing") {
    NaiveBayesClassifier naive_bayes_model;
    naive_bayes_model.Train(data_converter, 10);
    pixel_probabilities = naive_bayes_model.GetConditionalProbability();
    REQUIRE(pixel_probabilities[14][16][2][4] == Approx(0.375).epsilon(0.01));
    REQUIRE(pixel_probabilities[12][10][0][4] == Approx(0.344).epsilon(0.01));
    REQUIRE(pixel_probabilities[11][9][1][4] == Approx(0).epsilon(0.01));
    REQUIRE(pixel_probabilities[8][16][2][4] == Approx(0).epsilon(0.01));
  }
}

TEST_CASE("Processing for arbitrary image sizes") {
  string data_file_path = "data/test_smaller_images.txt";
  std::ifstream input_file(data_file_path);
  DataConverter data_converter;
  data_converter << input_file;
  map<int, double> image_class_probabilities;
  vector<vector<vector<vector<double>>>> pixel_probabilities;

  SECTION("Default Processor") {
    NaiveBayesClassifier naive_bayes_model;
    naive_bayes_model.Train(data_converter);
    image_class_probabilities = naive_bayes_model.GetPriorProbability();
    pixel_probabilities = naive_bayes_model.GetConditionalProbability();
    REQUIRE(image_class_probabilities[4] == Approx(0.667).epsilon(0.01));
    REQUIRE(pixel_probabilities[11][11][1][4] == Approx(1).epsilon(0.01));
    REQUIRE(pixel_probabilities[12][10][0][4] == Approx(1).epsilon(0.01));
    REQUIRE(pixel_probabilities[11][9][1][4] == Approx(0).epsilon(0.01));
    REQUIRE(pixel_probabilities[8][16][2][4] == Approx(1).epsilon(0.01));
  }

  SECTION("After laplace smoothing") {
    NaiveBayesClassifier naive_bayes_model;
    naive_bayes_model.Train(data_converter, 10);
    image_class_probabilities = naive_bayes_model.GetPriorProbability();
    pixel_probabilities = naive_bayes_model.GetConditionalProbability();
    REQUIRE(image_class_probabilities[1] == Approx(0.4782).epsilon(0.01));
    REQUIRE(pixel_probabilities[11][11][1][4] == Approx(0.375).epsilon(0.01));
    REQUIRE(pixel_probabilities[12][10][0][4] == Approx(0.375).epsilon(0.01));
    REQUIRE(pixel_probabilities[11][9][1][4] == Approx(0).epsilon(0.01));
    REQUIRE(pixel_probabilities[8][16][2][4] == Approx(0.375).epsilon(0.01));
  }
}

TEST_CASE("Writing trained model") {
  string data_file_path = "data/test_normal_images.txt";
  string output_file_path = "data/test_trained_model.txt";
  std::ifstream input_file(data_file_path);
  std::ofstream output_file(output_file_path);
  
  DataConverter data_converter;
  data_converter << input_file;
  NaiveBayesClassifier naive_bayes_model;
  naive_bayes_model.Train(data_converter, 10);
  naive_bayes_model >> output_file;

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
  NaiveBayesClassifier naive_bayes_model;
  std::ifstream input_file("data/naive_bayes_model.txt");
  naive_bayes_model << input_file;

  map<int, double> image_class_probabilities;
  vector<vector<vector<vector<double>>>> pixel_probabilities;
  image_class_probabilities = naive_bayes_model.GetPriorProbability();
  pixel_probabilities = naive_bayes_model.GetConditionalProbability();
 
  REQUIRE(image_class_probabilities[4] == Approx(0.1069).epsilon(0.001));
  REQUIRE(pixel_probabilities[11][11][1][4] == Approx(0.1914).epsilon(0.001));
  REQUIRE(pixel_probabilities[12][10][0][4] == Approx(0.297).epsilon(0.01));
  REQUIRE(pixel_probabilities[11][9][1][4] == Approx(0.283).epsilon(0.01));
  REQUIRE(pixel_probabilities[8][16][2][4] == Approx(0.149).epsilon(0.01));
}

TEST_CASE("Classify one image") {
  NaiveBayesClassifier naive_bayes_model;
  DataConverter data_converter;

  std::ifstream input_file("data/naive_bayes_model.txt");
  std::ifstream test_file("data/test_normal_images.txt");
  
  data_converter << test_file;
  naive_bayes_model << input_file;
  
  size_t result = naive_bayes_model.Classify(
      data_converter.GetDataset()[0].GetImageVector());
  
  std::vector<double> results(10);
  for (auto & prior_probability : naive_bayes_model.GetPriorProbability()) {
    double score = log(prior_probability.second);

    for (size_t i = 0; i < data_converter.GetImageSize(); i++) {
      for (size_t j = 0; j < data_converter.GetImageSize(); j++) {
        WrittenNumber::PixelColor pixelColor = 
            data_converter.GetDataset()[0].GetImageVector()[i][j];
        score += log(naive_bayes_model.
                     GetConditionalProbability()[i][j][static_cast<size_t>(
                         pixelColor)][prior_probability.first]);
      }
    }
    results[prior_probability.first] = score;
  }
  
  for (size_t i = 0; i < 10; i++) {
    REQUIRE(results[result] >= results[i]);
  }
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
