#pragma once

#include <map>
#include <sstream>

#include "core/data_converter.h"
#include "written_number.h"

namespace naivebayes {

using std::vector;

/**
 * Process the data to get probabilities for P(class = c) and
 * P(F_{i, j} = f | class = c).
 */
class DataProcessor {
public:
  /**
   * Default constructor used before loading model from a file.
   */
  DataProcessor();
  
  /**
   * Constructor for processing the dataset to get the probabilities info.
   * @param data_converter a data converter containing preprocessed dataset
   * @param laplace_parameter the laplace smoothing parameter for processing
   */
  DataProcessor(const DataConverter& data_converter,
                double laplace_parameter = 0);

  /**
   * Gets a map containing P(class = c) data.
   * @return a map(class = c, P(class = c))
   */
  const std::map<int, double>& GetClassProbability() const;

  /**
   * Gets a 4D vector containing P(F_{i, j} = f | class = c) data.
   * @return a vector consisting of P(F_{i, j} = f | class = c)
   */
  const vector<vector<vector<vector<double>>>>& GetPixelProbability() const;
  
  /**
   * Writes the prior probabilities P(class = c) and 
   * conditional probabilities P(F_{i, j} = f | class = c) to a file,
   * namely trained model.
   * @param output_file the file to write to
   * @param data_processor the data processor itself
   * @return output file after writing
   */
  friend std::ostream &operator<<(std::ostream& output_file,
                                  DataProcessor& data_processor);

  /**
   * 
   * @param input_file 
   * @param data_processor 
   * @return 
   */
  friend std::istream &operator>>(std::istream& input_file,
                                  DataProcessor& data_processor);
  
private:
  /**
   * Counts the number of each written number class in the dataset.
   * @param data_converter a data converter containing preprocessed dataset
   */
  void CountClasses(const DataConverter& data_converter);
  
  /**
   * Calculates P(class = c) for each written number class.
   * @param data_converter a data converter containing preprocessed dataset
   */
  void CalculateProbabilityForClasses(const DataConverter& data_converter);
  
  /**
   * Initiates each P(F_{i, j} = f | class = c) with 
   * k / (pixel_color_count * k + # classes belonging to class c).
   * @param data_converter a data converter containing preprocessed dataset
   */
  void InitiatePixelProbabilities(const DataConverter& data_converter);
  
  /**
   * Calculates P(F_{i, j} = f | class = c) for each point and class.
   * @param data_converter a data converter containing preprocessed dataset
   */
  void CalculateProbabilityForPixels(const DataConverter& data_converter);
  
  /**
   * Splits a line of string data to several substring data.
   * @param line a line in the trained model file
   * @return a list of substrings consisting of data a in line
   */
  std::vector<std::string> SplitDataStrings(const std::string& line, 
                                            char splitter);

  // Parameter k for laplace smoothing
  double laplace_parameter_;

  // The number of each class in the dataset
  std::map<int, size_t> class_count_;

  // A map of P(class = c)
  std::map<int, double> class_probabilities_;

  // A vector of P(F_{i, j} = f | class = c)
  vector<vector<vector<vector<double>>>> pixel_probabilities_;
};

} // namespace naivebayes

