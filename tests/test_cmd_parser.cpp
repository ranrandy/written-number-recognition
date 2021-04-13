#include <catch2/catch.hpp>

#include "core/cmd_parser.h"

using naivebayes::CmdParser;
using std::string;

TEST_CASE("Using command line to run") {
  char* argv_0 = const_cast<char *>(
      "/Users/lirunfeng/cinder_master/my-projects/naive-bayes-ranrandy/"
      "cmake-build-debug/train-model");
  
  char* name_read = const_cast<char *>("--read");
  char* name_load = const_cast<char *>("--load");
  char* name_train = const_cast<char *>("--train");
  char* name_test = const_cast<char *>("--test");
  char* name_naive_bayes_k = const_cast<char *>("--naive_bayes_k");
  char* name_nearest_k = const_cast<char *>("--nearest_k");
  char* flag_save = const_cast<char *>("-s");

  char* data_file_path = const_cast<char *>("data/trainingimagesandlabels.txt");
  char* test_file_path = const_cast<char *>("data/testimagesandlabels.txt");
  char* save_path = const_cast<char *>("data/naive_bayes_model.txt");
  char* model_path = const_cast<char *>("data/naive_bayes_model.txt");
  char* no_file_path = const_cast<char *>("no_file");
  
  char* naive_bayes = const_cast<char *>("naive_bayes");
  char* k_nearest_neighbor = const_cast<char *>("knn");
  // char* artificial_neural_network = const_cast<char *>("ann");
  // char* decision_tree = const_cast<char *>("decision_tree");
  // char* gaussian_naive_bayes = const_cast<char *>("g_naive_bayes");
  // char* boosting = const_cast<char *>("boosting");
  
  char* naive_bayes_k = const_cast<char *>("0.9");
  char* nearest_k = const_cast<char *>("15");
  char* naive_bayes_k_default = const_cast<char *>("0");
  char* nearest_k_default = const_cast<char *>("10");

  SECTION("Read and train the dataset using naive bayes algorithm") {
    char* argv[5] = {argv_0,
                     name_read, data_file_path,
                     name_train, naive_bayes};
    CmdParser cmd_parser(5, argv);
    REQUIRE(cmd_parser.GetDatasetFilePath() == data_file_path);
    REQUIRE(cmd_parser.GetTestFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetModelFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetSaveFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetAlgorithm() == naive_bayes);
    REQUIRE(cmd_parser.GetNaiveBayesK() == std::stod(naive_bayes_k_default));
    REQUIRE(cmd_parser.GetNearestK() == std::stoi(nearest_k_default));
  }
  
  SECTION("Read and train the dataset using naive bayes algorithm, "
          "then save the model to a file") {
    char* argv[7] = {argv_0, 
                     name_read, data_file_path, 
                     name_train, naive_bayes, 
                     flag_save, save_path};
    CmdParser cmd_parser(7, argv);
    REQUIRE(cmd_parser.GetDatasetFilePath() == data_file_path);
    REQUIRE(cmd_parser.GetTestFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetModelFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetSaveFilePath() == save_path);
    REQUIRE(cmd_parser.GetAlgorithm() == naive_bayes);
    REQUIRE(cmd_parser.GetNaiveBayesK() == std::stod(naive_bayes_k_default));
    REQUIRE(cmd_parser.GetNearestK() == std::stoi(nearest_k_default));
  }

  SECTION("Read and train the dataset using naive bayes algorithm and "
          "laplace parameter k = 0.9, then save the model to a file") {
    char* argv[9] = {argv_0,
                     name_read, data_file_path,
                     name_train, naive_bayes,
                     name_naive_bayes_k, naive_bayes_k,
                     flag_save, save_path, };
    CmdParser cmd_parser(9, argv);
    REQUIRE(cmd_parser.GetDatasetFilePath() == data_file_path);
    REQUIRE(cmd_parser.GetTestFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetModelFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetSaveFilePath() == save_path);
    REQUIRE(cmd_parser.GetAlgorithm() == naive_bayes);
    REQUIRE(cmd_parser.GetNaiveBayesK() == std::stod(naive_bayes_k));
    REQUIRE(cmd_parser.GetNearestK() == std::stoi(nearest_k_default));
  }

  SECTION("Load a naive bayes model from a file") {
    char* argv[5] = {argv_0,
                     name_load, model_path,
                     name_train, naive_bayes};
    CmdParser cmd_parser(5, argv);
    REQUIRE(cmd_parser.GetDatasetFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetTestFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetModelFilePath() == model_path);
    REQUIRE(cmd_parser.GetSaveFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetAlgorithm() == naive_bayes);
    REQUIRE(cmd_parser.GetNaiveBayesK() == std::stod(naive_bayes_k_default));
    REQUIRE(cmd_parser.GetNearestK() == std::stoi(nearest_k_default));
  }

  SECTION("Load a naive bayes model from a file and use the model to test") {
    char* argv[7] = {argv_0,
                     name_load, model_path,
                     name_train, naive_bayes, 
                     name_test, test_file_path};
    CmdParser cmd_parser(7, argv);
    REQUIRE(cmd_parser.GetDatasetFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetTestFilePath() == test_file_path);
    REQUIRE(cmd_parser.GetModelFilePath() == model_path);
    REQUIRE(cmd_parser.GetSaveFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetAlgorithm() == naive_bayes);
    REQUIRE(cmd_parser.GetNaiveBayesK() == std::stod(naive_bayes_k_default));
    REQUIRE(cmd_parser.GetNearestK() == std::stoi(nearest_k_default));
  }

  SECTION("Read and train the dataset using naive bayes algorithm, "
          "then use the trained model to test") {
    char* argv[9] = {argv_0,
                     name_read, data_file_path,
                     name_train, naive_bayes,
                     name_naive_bayes_k, naive_bayes_k,
                     name_test, test_file_path};
    CmdParser cmd_parser(9, argv);
    REQUIRE(cmd_parser.GetDatasetFilePath() == data_file_path);
    REQUIRE(cmd_parser.GetTestFilePath() == test_file_path);
    REQUIRE(cmd_parser.GetModelFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetSaveFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetAlgorithm() == naive_bayes);
    REQUIRE(cmd_parser.GetNaiveBayesK() == std::stod(naive_bayes_k));
    REQUIRE(cmd_parser.GetNearestK() == std::stoi(nearest_k_default));
  }
  
  SECTION("Load a dataset to classify images using k nearest neighbor "
          "algorithm") {
    char* argv[9] = {argv_0,
                     name_load, data_file_path,
                     name_train, k_nearest_neighbor,
                     name_nearest_k, nearest_k,
                     name_test, test_file_path};
    CmdParser cmd_parser(9, argv);
    REQUIRE(cmd_parser.GetDatasetFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetTestFilePath() == test_file_path);
    REQUIRE(cmd_parser.GetModelFilePath() == data_file_path);
    REQUIRE(cmd_parser.GetSaveFilePath() == no_file_path);
    REQUIRE(cmd_parser.GetAlgorithm() == k_nearest_neighbor);
    REQUIRE(cmd_parser.GetNaiveBayesK() == std::stod(naive_bayes_k_default));
    REQUIRE(cmd_parser.GetNearestK() == std::stoi(nearest_k));
  }
}