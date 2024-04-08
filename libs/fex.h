#pragma once

#include "logger.h"

#include <vector>
#include <string>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <thread>
#include <fstream>
#include <iostream>
#include <stdint.h>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;
    bool valuesRead = false;

    std::vector<float> data;
};

struct whisper_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

class FeatureExtractor {
public:
    whisper_filters mel_filters_;
    FeatureExtractor();

    /**
     * Run the extractor on the given std::vector of uint8_t
     * and return a corresponding 3D vector of features
    */
    std::vector<std::vector<std::vector<float>>> run(const std::vector<int16_t>& data);

private:
    /**
     * By default, a data directory containing filters file
     * is assumed to be in the same top-level directory
     */
    std::string whisperFiltersFileName_ = "data/mel_filters.bin";
};