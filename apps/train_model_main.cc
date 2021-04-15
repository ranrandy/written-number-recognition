#include "core/cmd_parser.h"
#include "core/dataset.h"
#include "core/k_nearest_neighbors_classifier.h"
#include "core/naive_bayes_classifier.h"
#include "core/written_number.h"

using namespace naivebayes;

int main(int argc, char* argv[]) {
  CmdParser cmd_parser(argc, argv);
  
  // Trains the dataset.
  if (cmd_parser.GetDatasetFilePath() != cmd_parser.kNoFile) {
    Dataset data_converter;
    std::ifstream dataset_file_path(cmd_parser.GetDatasetFilePath());
    
    data_converter << dataset_file_path;
    
    if (cmd_parser.GetAlgorithm() == cmd_parser.kNaiveBayes && 
        dataset_file_path.is_open()) {
      NaiveBayesClassifier nb_model;
      nb_model.Train(data_converter, cmd_parser.GetNaiveBayesK());
      
      if (cmd_parser.GetSaveFilePath() != cmd_parser.kNoFile) {
        std::ofstream model_file(cmd_parser.GetSaveFilePath());
        
        if (model_file.is_open()) {
          nb_model >> model_file;
        }
        model_file.close();
      }

      if (cmd_parser.GetTestFilePath() != cmd_parser.kNoFile) {
        std::ifstream test_dataset_file_path(cmd_parser.GetTestFilePath());

        if (test_dataset_file_path.is_open()) {
          Dataset test_data_converter;
          test_data_converter << test_dataset_file_path;
          double model_accuracy = 
              nb_model.EvaluateAccuracy(test_data_converter);
          std::cout << "The accuracy of the " << cmd_parser.GetAlgorithm() <<
                    " model is " << model_accuracy << std::endl;
          nb_model.OutputConfusingMatrix();
        }
        test_dataset_file_path.close();
      }
    }
    dataset_file_path.close();
  }
  
  // Loads the model / dataset (and tests the test dataset).
  if (cmd_parser.GetModelFilePath() != cmd_parser.kNoFile) {
    std::ifstream model_file_path(cmd_parser.GetModelFilePath());
    
    // Classify (Naive Bayes)
    if (cmd_parser.GetAlgorithm() == cmd_parser.kNaiveBayes && 
        model_file_path.is_open()) {
      NaiveBayesClassifier nb_model;
      nb_model << model_file_path;
      
      if (cmd_parser.GetTestFilePath() != cmd_parser.kNoFile) {
        std::ifstream test_dataset_file_path(cmd_parser.GetTestFilePath());
        
        if (test_dataset_file_path.is_open()) {
          Dataset data_converter;
          data_converter << test_dataset_file_path;
          double model_accuracy = nb_model.EvaluateAccuracy(data_converter);
          std::cout << "The accuracy of the " << cmd_parser.GetAlgorithm() <<
                    " model is " << model_accuracy << std::endl;
          nb_model.OutputConfusingMatrix();
        }
        test_dataset_file_path.close();
      }
    }
    
    // Classify (K Nearest Neighbors)
    if (cmd_parser.GetAlgorithm() == cmd_parser.kKNearestNeighbor && 
        model_file_path.is_open()) {
      KNearestNeighborClassifier knn_model;
      Dataset dataset_converter;
      dataset_converter << model_file_path;
      
      if (cmd_parser.GetTestFilePath() != cmd_parser.kNoFile) {
        std::ifstream test_dataset_file_path(cmd_parser.GetTestFilePath());

        if (test_dataset_file_path.is_open()) {
          Dataset test_data_converter;
          test_data_converter << test_dataset_file_path;
          double model_accuracy = 
              knn_model.EvaluateAccuracy(test_data_converter, 
                                         dataset_converter, 
                                         cmd_parser.GetNearestK());
          std::cout << "The accuracy of the " << cmd_parser.GetAlgorithm() <<
                    " model is " << model_accuracy << std::endl;
          knn_model.OutputConfusingMatrix();
        }
        test_dataset_file_path.close();
      }
    }
    model_file_path.close();
  }
  return 0;
}
