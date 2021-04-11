#pragma once

#include "tclap/CmdLine.h"

namespace naivebayes {

using TCLAP::CmdLine;
using std::string;

/**
 * Parse the command line input to run the project.
 */
class CmdParser {
public:
  const string kNaiveBayes = "naive_bayes";
  const string kKNearestNeighbor = "knn";
  const string kArtificialNeuralNetwork = "ann";
  const string kDecisionTree = "decision_tree";
  const string kGaussianNaiveBayes = "g_naive_bayes";
  const string kVotingBoosting = "voting_boosting";
  const string kNoFile = "no_file";
  
  /**
   * Constructor for CmdParser. Create the parsing methods and parse the 
   * command line input.
   * @param argc the number of parameters from command line
   * @param argv the parameters from command line
   */
  CmdParser(int argc, char* argv[]);
  
  /**
   * Gets the dataset file path.
   * @return dataset file path
   */
  const string& GetDatasetFilePath() const;
  
  /**
   * Gets the test file path.
   * @return test file path
   */
  const string& GetTestFilePath() const;
  
  /**
   * Gets the model file path
   * @return model file path
   */
  const string& GetModelFilePath() const;
  
  /**
   * Gets the path where model will be saved to.
   * @return the path where model will be saved to
   */
  const string& GetSaveFilePath() const;
  
  /**
   * Gets the algorithm that will be used to train the dataset.
   * @return the algorithm that will be used to train the dataset
   */
  const string& GetAlgorithm() const;
  
  /**
   * Gets the laplace smoothing parameter k for naive bayes algorithm.
   * @return the laplace smoothing parameter k for naive bayes algorithm
   */
  double GetNaiveBayesK() const;
  
  /**
   * Gets the parameter for k nearest neighbors algorithm.
   * @return the parameter for k nearest neighbors algorithm.
   */
  double GetNearestK() const;

private:
  const string kProjectVersion = "0.1";
  const string kStringReturnType = "string";
  const char kDelimiter = ' ';

  string dataset_file_path_;
  string test_file_path_;
  string model_file_path_;
  string save_file_path_;
  string algorithm_;
  
  double naive_bayes_k_;
  double nearest_k_;
};

} // namespace naivebayes

