#pragma once

#include <map>
#include <set>

#include "written_number.h"

namespace naivebayes {

class DataProcessor {
public:
  DataProcessor(const std::vector<WrittenNumber>& written_numbers);

  const std::map<std::string, double>& GetClassProbability() const;

private:
  void CalculateProbabilityForClasses(
      const std::vector<WrittenNumber>& written_numbers);

  std::map<std::string, double> image_class_probabilities_;
};

}

