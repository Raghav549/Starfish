#pragma once
#include <cstdint>
#include <vector>
namespace starfish {
int crossing_number(const uint8_t* pixels,int width,int height,int x,int y);
std::vector<int> crossing_numbers(const uint8_t* pixels,int width,int height);
}