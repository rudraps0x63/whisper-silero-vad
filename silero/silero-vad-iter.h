#pragma once

#include "wav.h"

#include <vector>
#include <string>

std::vector<int16_t> WriteToWAVVector(const std::string& modelPath, std::vector<float>& wavData);