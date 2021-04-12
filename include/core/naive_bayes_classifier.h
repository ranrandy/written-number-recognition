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
class NaiveBayesClassifier {
public:
  typedef vector<vector<vector<vector<double>>>> vec4;
  typedef vector<vector<vector<double>>> vec3;
  typedef vector<vector<double>> vec2;
  
  /**
   * Default constructor used before loading model from a file.
   */
  NaiveBayesClassifier();
  
  /**
   * Constructor for processing the dataset to get the probabilities info.
   * @param data_converter a data converter containing preprocessed dataset
   * @param laplace_parameter the laplace smoothing parameter for processing
   */
  NaiveBayesClassifier(const DataConverter& data_converter,
                double laplace_parameter = 0);

  /**
   * Test with the testing dataset to obtain the accuracy of the model.
   * @param data_converter a data converter containing preprocessed dataset 
   * @return the accuracy of the model
   */
  double EvaluateAccuracy(const DataConverter& data_converter);
  
  /**
   * Classify one hand written number.
   * @param image_vector a 2D vector representing pixels of a written number
   * @return the classification result of this written number
   */
  size_t Classify(const vector<vector<WrittenNumber::PixelColor>>& 
                      image_vector);
  
  /**
   * Gets a map containing P(class = c) data.
   * @return a map(class = c, P(class = c))
   */
  const std::map<int, double>& GetPriorProbability() const;

  /**
   * Gets a 4D vector containing P(F_{i, j} = f | class = c) data.
   * @return a vector consisting of P(F_{i, j} = f | class = c)
   */
  const vec4& GetConditionalProbability() const;
  
  /**
   * Writes the prior probabilities P(class = c) and 
   * conditional probabilities P(F_{i, j} = f | class = c) to a file,
   * namely trained model.
   * @param output_file the file to write to
   * @return output file after writing
   */
  std::ostream& operator>>(std::ostream& output_file);

  /**
   * Reads the prior probabilities P(class = c) and 
   * conditional probabilities P(F_{i, j} = f | class = c) from a file,
   * namely the trained model.
   * @param input_file the file to read
   * @return input file after reading
   */
  std::istream& operator<<(std::istream& input_file);
  
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
    * Initiates each P(F_{i, j} = f | class = c) with 0.
    * @param image_size the size of one image in the dataset 
    * @param color_count the number of color that will be in one image
    * @param max_number the greatest written number in the dataset
    */ 
  void InitiatePixelProbabilities(size_t image_size, 
                                  size_t color_count, 
                                  size_t max_number);
  
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
  // pixel_probabilities_[i][j][f][c]
  vec4 pixel_probabilities_;
};

} // namespace naivebayes

