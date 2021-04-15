#include <catch2/catch.hpp>

#include "visualizer/sketchpad.h"
#include "core/written_number.h"

using naivebayes::visualizer::Sketchpad;
using naivebayes::WrittenNumber;

TEST_CASE("Handle brush") {
  std::vector<WrittenNumber::PixelColor> 
      row_shaded_pixels(28, WrittenNumber::PixelColor::kWhite);
  std::vector<std::vector<WrittenNumber::PixelColor>> 
      shaded_pixels(28, row_shaded_pixels);

  SECTION("Corner") {
    Sketchpad sketchpad(glm::vec2(100, 100), 28, 875 - 2 * 100);
    glm::vec2 brush_screen_coords(101, 101);
    sketchpad.HandleBrush(brush_screen_coords);

    shaded_pixels[0][0] = WrittenNumber::PixelColor::kBlack;
    REQUIRE(shaded_pixels == sketchpad.GetShadedPixels());
  }

  SECTION("Center") {
    Sketchpad sketchpad(glm::vec2(100, 100), 28, 875 - 2 * 100);
    glm::vec2 brush_screen_coords(148, 148);
    sketchpad.HandleBrush(brush_screen_coords);

    shaded_pixels[1][1] = WrittenNumber::PixelColor::kBlack;
    shaded_pixels[1][2] = WrittenNumber::PixelColor::kBlack;
    shaded_pixels[2][1] = WrittenNumber::PixelColor::kBlack;
    shaded_pixels[2][2] = WrittenNumber::PixelColor::kBlack;
    REQUIRE(shaded_pixels == sketchpad.GetShadedPixels());
  }
}

TEST_CASE("Clear") {
  Sketchpad sketchpad(glm::vec2(100, 100), 28, 875 - 2 * 100);
  glm::vec2 brush_screen_coords(101, 101);
  sketchpad.HandleBrush(brush_screen_coords);
  sketchpad.Clear();
  
  for (size_t i = 0; i < 28; i++) {
    for (size_t j = 0; j < 28; j++) {
      REQUIRE(sketchpad.GetShadedPixels()[i][j] == 
              WrittenNumber::PixelColor::kWhite);
    }
  }
}
