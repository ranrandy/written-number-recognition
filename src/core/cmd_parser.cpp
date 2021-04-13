#include "core/cmd_parser.h"

namespace naivebayes {

using TCLAP::CmdLine;
using TCLAP::ValueArg;
using TCLAP::ValuesConstraint;
using std::vector;
using std::string;

CmdParser::CmdParser(int argc, char* argv[]) {
  try {
    CmdLine cmd("Command description message", kDelimiter, kProjectVersion);

    // Specify the dataset file to read from 
    // or the model to load from
    ValueArg<string> read_arg("", "read", "read a dataset", false, kNoFile,
                              kStringReturnType);
    ValueArg<string> load_arg("", "load", "load a model", false, kNoFile,
                              kStringReturnType);
    cmd.xorAdd(read_arg, load_arg);
    
    // Specify the parameter for an algorithm
    ValueArg<double> nb_k_arg("", "naive_bayes_k", 
                              "laplace parameter for naive bayes "
                              "algorithm", false, 0, "double");
    cmd.add(nb_k_arg);

    // Specify the parameter for an algorithm
    ValueArg<size_t> nearest_k_arg("", "nearest_k",
                                   "parameter k for k nearest neighbor "
                                   "algorithm", false, 10, "size_t");
    cmd.add(nearest_k_arg);

    // Specify whether to save the trained model to a file or not
    ValueArg<string> save_arg("s", "save", "save to file", false, kNoFile,
                              kStringReturnType);
    cmd.add(save_arg);

    // Specify a test file to test
    ValueArg<string> test_arg("", "test", "test a dataset", false, kNoFile,
                              kStringReturnType);
    cmd.add(test_arg);

    // Specify the algorithm to train the dataset with
    vector<string> models;
    models.emplace_back(kNaiveBayes);
    models.emplace_back(kKNearestNeighbor);
    models.emplace_back(kGaussianNaiveBayes);
    models.emplace_back(kDecisionTree);
    models.emplace_back(kVotingBoosting);
    models.emplace_back(kArtificialNeuralNetwork);
    ValuesConstraint<string> model_names(models);
    ValueArg<string> algorithm_arg("", "train", "algorithm to train with",
                                   true, kNaiveBayes, &model_names);
    cmd.add(algorithm_arg);
    
    // Parse each block of the entire command line input
    cmd.parse(argc, argv);
    
    dataset_file_path_ = read_arg.getValue();
    test_file_path_ = test_arg.getValue();
    model_file_path_ = load_arg.getValue();
    save_file_path_ = save_arg.getValue();
    algorithm_ = algorithm_arg.getValue();
    naive_bayes_k_ = nb_k_arg.getValue();
    nearest_k_ = nearest_k_arg.getValue();
  } catch (TCLAP::ArgException &e) {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << 
        std::endl;
  }
}

const string& CmdParser::GetDatasetFilePath() const {
  return dataset_file_path_;
}

const string& CmdParser::GetTestFilePath() const {
  return test_file_path_;
}

const string& CmdParser::GetModelFilePath() const {
  return model_file_path_;
}

const string& CmdParser::GetSaveFilePath() const {
  return save_file_path_;
}

const string& CmdParser::GetAlgorithm() const {
  return algorithm_;
}
 
double CmdParser::GetNaiveBayesK() const {
  return naive_bayes_k_;
}

double CmdParser::GetNearestK() const {
  return nearest_k_;
}

} // naivebayes