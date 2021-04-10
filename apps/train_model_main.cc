#include "core/cmd_parser.h"
#include "core/data_converter.h"
#include "core/naive_bayes_model.h"
#include "core/written_number.h"

using namespace naivebayes;

int main(int argc, char* argv[]) {
  CmdParser cmd_parser(argc, argv);
  
  // Trains the dataset.
  if (cmd_parser.GetDatasetFilePath() != cmd_parser.kNoFile) {
    DataConverter data_converter;
    std::ifstream dataset_file_path(cmd_parser.GetDatasetFilePath());
    
    data_converter << dataset_file_path;
    
    if (cmd_parser.GetAlgorithm() == cmd_parser.kNaiveBayes && 
        dataset_file_path.is_open()) {
      NaiveBayesModel nb_model(data_converter, cmd_parser.GetNaiveBayesK());
      
      if (cmd_parser.GetSaveFilePath() != cmd_parser.kNoFile) {
        std::ofstream model_file(cmd_parser.GetSaveFilePath());
        
        if (model_file.is_open()) {
          nb_model >> model_file;
        }
        model_file.close();
      }
    }
    dataset_file_path.close();
  }
  
  // Loads the model (and tests the test dataset).
  if (cmd_parser.GetModelFilePath() != cmd_parser.kNoFile) {
    std::ifstream model_file_path(cmd_parser.GetModelFilePath());
    
    if (cmd_parser.GetAlgorithm() == cmd_parser.kNaiveBayes && 
        model_file_path.is_open()) {
      NaiveBayesModel nb_model;
      nb_model << model_file_path;
      
      if (cmd_parser.GetTestFilePath() != cmd_parser.kNoFile) {
        std::ifstream test_dataset_file_path(cmd_parser.GetTestFilePath());
        
        if (test_dataset_file_path.is_open()) {
          DataConverter data_converter;
          data_converter << test_dataset_file_path;
          double model_accuracy = nb_model.EvaluateAccuracy(data_converter);
          std::cout << "The accuracy of the " << cmd_parser.GetAlgorithm() <<
                    " model is " << model_accuracy << std::endl;
        }
        test_dataset_file_path.close();
      }
    }
    model_file_path.close();
  }
  return 0;
}
