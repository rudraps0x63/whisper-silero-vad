#pragma once

/*  Copyright (c) 2023 by Contributors
 * \file cli_main.cc
 * \brief Implementation of a CLI version of chat
 */

// NOTE we only interact with the module through tvm runtime
// so there is no need to depend on a header interface
// the same set of operations can be implemented in other languages
#define PICOJSON_USE_INT64
#define __STDC_FORMAT_MACROS

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <unistd.h>

#include <bitset>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "llm_chat.h"
#include "picojson.h"
#include "translate.h"

struct ModelPaths {
  /*!
   * \brief Path to mlc-llm-config.json
   */
  std::filesystem::path config;
  /*!
   * \brief Path to ndarray-cache.json
   */
  std::filesystem::path params;
  /*!
   * \brief Path to ${model}-${device}.{so|dylib}
   *
   * This dynamic library contains all the compute kernels used in LLM inference, and can be
   * loaded using tvm::runtime::Module::LoadFromFile.
   */
  std::filesystem::path lib;

  static ModelPaths Find(const std::string& device_name, const std::string& local_id, const std::string& user_lib_path);
};

class TranslateModule {
 public:
  /*!
   * \brief Constructor
   * \param device the device to run the chat on.
   */
  explicit TranslateModule(const DLDevice& device);
  /*!
   * \brief Reload the module to a new model path.
   * \param model The model path spec.
   */
  void Reload(const ModelPaths& model);

  /*! \return A text describing the runtime statistics. */
  std::string RuntimeStatsText();

  /*! \return A text describing the token statistics. */
  std::string VerboseRuntimeStatsText();

  /*!
   * \brief Run prefill stage for a given input and decode the first output token.
   * \param input the user input.
   */
  void Prefill(const std::string& input);

  /*!
   * \brief Run one decode step to decode the next token.
   */
  void Decode();

  /*
   * \brief Run one encode step to encode the input text
   */
  void Encode(const std::string& input);

 protected:
  // TVM Modules and functions with TVM's calling convention
  tvm::runtime::Module chat_mod_;
  tvm::runtime::PackedFunc encode_;
  tvm::runtime::PackedFunc prefill_;
  tvm::runtime::PackedFunc decode_;
  tvm::runtime::PackedFunc reload_;
  tvm::runtime::PackedFunc runtime_stats_text_;
  tvm::runtime::PackedFunc verbose_runtime_stats_text_;

  std::string lib_path_;
  tvm::runtime::Module executable_;
};

class Translator {
public:
  tvm::runtime::Module translate_mod;
  std::string model_path;

  Translator(const std::vector<std::string>& argv);

  std::string generateFromFeatureVector(const std::vector<std::vector<std::vector<float>>>& data);
  std::string generateFromFeatureFile(const std::string& fileName);

private:
  std::string generateTranscription(const std::vector<float>& input_features);
};
