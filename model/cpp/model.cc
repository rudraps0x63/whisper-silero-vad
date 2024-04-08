#include "model.h"

const std::vector<std::string> quantization_presets = {"q3f16_0",  //
                                                       "q4f16_0",  //
                                                       "q4f16_1",  //
                                                       "q4f16_2",  //
                                                       "q4f32_0",  //
                                                       "q8f16_0",  //
                                                       "q0f16",    //
                                                       "q0f32"};

std::pair<std::string, int> DetectDevice(std::string device) {
  using tvm::runtime::DeviceAPI;

  std::string device_name;
  int device_id;
  int delimiter_pos = device.find(":");

  if (delimiter_pos == std::string::npos) {
    device_name = device;
    device_id = 0;
  } else {
    device_name = device.substr(0, delimiter_pos);
    device_id = std::stoi(device.substr(delimiter_pos + 1, device.length()));
  }

  if (device_name == "auto") {
    bool allow_missing = true;
    if (DeviceAPI::Get(DLDevice{kDLCUDA, 0}, allow_missing)) {
      return {"cuda", device_id};
    }
    if (DeviceAPI::Get(DLDevice{kDLMetal, 0}, allow_missing)) {
      return {"metal", device_id};
    }
    if (DeviceAPI::Get(DLDevice{kDLROCM, 0}, allow_missing)) {
      return {"rocm", device_id};
    }
    if (DeviceAPI::Get(DLDevice{kDLVulkan, 0}, allow_missing)) {
      return {"vulkan", device_id};
    }
    if (DeviceAPI::Get(DLDevice{kDLOpenCL, 0}, allow_missing)) {
      return {"opencl", device_id};
    }
    // TODO: Auto-detect devices for mali
    LOG(FATAL) << "Cannot auto detect device-name";
  }
  return {device_name, device_id};
}

DLDevice GetDevice(const std::string& device_name, int device_id) {
  if (device_name == "cuda") return DLDevice{kDLCUDA, device_id};
  if (device_name == "metal") return DLDevice{kDLMetal, device_id};
  if (device_name == "rocm") return DLDevice{kDLROCM, device_id};
  if (device_name == "vulkan") return DLDevice{kDLVulkan, device_id};
  if (device_name == "opencl" || device_name == "mali") return DLDevice{kDLOpenCL, device_id};
  LOG(FATAL) << "Invalid device name: " << device_name
             << ". Please enter the device in the form 'device_name:device_id'"
                " or 'device_name', where 'device_name' needs to be one of 'cuda', 'metal', "
                "'vulkan', 'rocm', 'opencl', 'auto'.";
  return DLDevice{kDLCPU, 0};
}

/*!
 * \brief Search for file path return the first result.
 *
 * \param search_paths The paths to search for the file.
 * \param names The names of to look for.
 * \param suffixes The suffix to look for.
 */
std::optional<std::filesystem::path> FindFile(
    const std::vector<std::filesystem::path>& search_paths,  //
    const std::vector<std::string>& names,                   //
    const std::vector<std::string>& suffixes) {
  for (const std::filesystem::path& prefix : search_paths) {
    for (const std::string& name : names) {
      for (const std::string& suffix : suffixes) {
        try {
          std::filesystem::path path = std::filesystem::canonical(prefix / (name + suffix));
          if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path)) {
            return path;
          }
        } catch (const std::filesystem::filesystem_error& e) {
        }
      }
    }
  }
  return std::nullopt;
}

/**
 * get default lib suffixes
 */
std::vector<std::string> GetLibSuffixes() {
#if defined(WIN32)
  return {".dll"};
#elif defined(__APPLE__)
  return {".dylib", ".so"};
#else
  return {".so"};
#endif
}

std::string GetArchSuffix() {
#if defined(__x86_64__)
  return "_x86_64";
#elif defined(__aarch64__)
  return "_arm64";
#endif
  return "";
}

void PrintSpecialCommands()
{
  std::cout << "You can use the following special commands:\n"
            << "  /help               print the special commands\n"
            << "  /exit               quit the cli\n"
            << "  /stats              print out the latest stats (token/sec)\n"
            << "  /reset              restart a fresh chat\n"
            << "  /reload [model]  reload model `model` from disk, or reload the current "
               "model if `model` is not specified\n"
            << std::endl
            << std::flush;
}

/* class TranslateModule */
TranslateModule::TranslateModule(const DLDevice& device)
{
  this->chat_mod_ = mlc::llm::CreateTranslateModule(device);
  this->encode_ = this->chat_mod_->GetFunction("encode");
  this->prefill_ = this->chat_mod_->GetFunction("prefill");
  this->decode_ = this->chat_mod_->GetFunction("decode");
  this->reload_ = this->chat_mod_->GetFunction("reload");
  this->runtime_stats_text_ = this->chat_mod_->GetFunction("runtime_stats_text");
  this->verbose_runtime_stats_text_ = this->chat_mod_->GetFunction("verbose_runtime_stats_text");
  this->lib_path_ = "";
  this->executable_ = tvm::runtime::Module(nullptr);
  ICHECK(encode_ != nullptr);
  // ICHECK(prefill_ != nullptr);
  // ICHECK(decode_ != nullptr);
  ICHECK(reload_ != nullptr);
  ICHECK(runtime_stats_text_ != nullptr);
  ICHECK(verbose_runtime_stats_text_ != nullptr);
}

void TranslateModule::Reload(const ModelPaths& model)
{
  std::cout << "Loading model..." << std::endl;
  std::string new_lib_path = model.lib.string();
  std::string new_model_path = model.config.parent_path().string();
  if (this->lib_path_ != new_lib_path) {
    this->lib_path_ = new_lib_path;
  }
  reload_(tvm::runtime::String(lib_path_), tvm::runtime::String(new_model_path));
  std::cout << "Loading finished" << std::endl << std::flush;
}

std::string TranslateModule::RuntimeStatsText()
{
  return runtime_stats_text_();
}

std::string TranslateModule::VerboseRuntimeStatsText()
{
  return verbose_runtime_stats_text_();
}

void TranslateModule::Prefill(const std::string& input)
{
  prefill_(input);
}

void TranslateModule::Decode()
{
  decode_();
}

void TranslateModule::Encode(const std::string& input)
{
  encode_(input);
}
/* [end class] */


std::optional<std::filesystem::path> TryInferMLCChatConfig(const std::string& local_id) {
  return FindFile(
      {
          local_id,                              // full path, or just the name
          "dist/prebuilt/" + local_id,           // Using prebuilt workflow
          "dist/" + local_id + "/params",        // Default directory after mlc_llm.build_model()
          "dist/prebuilt/mlc-chat-" + local_id,  // Also prebuilt workflow, but missed prefix
      },
      {"mlc-chat-config"}, {".json"});
}

std::string ReadStringFromJSONFile(const std::filesystem::path& config_path,
                                   const std::string& key) {
  std::string config_json_str;
  {
    std::ifstream config_istream(config_path.string());
    ICHECK(config_istream);
    std::ostringstream config_ostream;
    config_ostream << config_istream.rdbuf();
    config_json_str = config_ostream.str();
  }
  // Parse MLC's config json to figure out where the model lib is
  picojson::value config_info;
  picojson::parse(config_info, config_json_str);
  auto config = config_info.get<picojson::object>();
  ICHECK(config[key].is<std::string>());
  return config[key].get<std::string>();
}


ModelPaths
ModelPaths::Find(const std::string& device_name,
                 const std::string& local_id,
                 const std::string& user_lib_path)
{
  // Step 1. Find config path
  std::filesystem::path config_path;
  if (auto path = TryInferMLCChatConfig(local_id)) {
    config_path = path.value();
  } else {
    LOG(FATAL)
        << "The model folder provided does not seem to refer to a valid mlc-llm model folder. "
           "Specifically, we cannot find `mlc-chat-config.json`, a required file. You should "
           "provide a path that contains the file. "
           "According to your input `"
        << local_id << "`, we looked at folder(s):\n"
        << "- " + local_id << "\n"
        << "- dist/prebuilt/" + local_id << "\n"
        << "- dist/" + local_id + "/params"
        << "\n"
        << "- dist/prebuilt/mlc-chat-" + local_id;
    exit(1);
  }
  std::cout << "Use MLC config: " << config_path << std::endl;
  // Step 2. Find parameters
  std::filesystem::path params_json;
  if (auto path = FindFile({config_path.parent_path().string()}, {"ndarray-cache"}, {".json"})) {
    params_json = path.value();
  } else {
    std::cerr << "Cannot find \"ndarray-cache.json\" for params: " << config_path.parent_path()
              << std::endl;
    exit(1);
  }
  std::cout << "Use model weights: " << params_json << std::endl;
  // Step 3. Find model lib path
  std::filesystem::path lib_path;
  if (!user_lib_path.empty()) {
    lib_path = user_lib_path;
    if (!std::filesystem::exists(lib_path) || !std::filesystem::is_regular_file(lib_path)) {
      LOG(FATAL) << "The `lib_path` you passed in is not a file: " << user_lib_path << "\n";
      exit(1);
    }
  } else {
    std::string lib_local_id = ReadStringFromJSONFile(config_path, "model_lib");
    std::string lib_name = lib_local_id + "-" + device_name;
    if (auto path = FindFile({lib_local_id,
                              "dist/prebuilt/lib",  // Using prebuilt workflow
                              "dist/" + local_id, "dist/prebuilt/" + lib_local_id},
                             {
                                 lib_name + GetArchSuffix(),
                                 lib_name,
                             },
                             GetLibSuffixes())) {
      lib_path = path.value();
    } else {
      LOG(FATAL) << "Cannot find the model library that corresponds to `" << lib_local_id << "`.\n"
                 << "We searched over the following possible paths: \n"
                 << "- " + lib_local_id << "\n"
                 << "- dist/prebuilt/lib \n"
                 << "- dist/" + local_id << "\n"
                 << "- dist/prebuilt/" + lib_local_id << "\n"
                 << "If you would like to directly specify the full model library path, you may "
                 << "consider passing in the `--model-lib-path` argument.\n";
      exit(1);
    }
  }
  std::cout << "Use model library: " << lib_path << std::endl;
  return ModelPaths{config_path, params_json, lib_path};
}


// Function to create a 3D NDArray from a 1D vector of float
tvm::runtime::NDArray
Create3DNDArrayFrom1DArray(const std::vector<float>& data_1d, int dim1, int dim2, int dim3)
{
  // Check if the total number of elements matches the product of dimensions
  if (data_1d.size() != dim1 * dim2 * dim3) {
    throw std::runtime_error("Dimension mismatch");
  }

  // Define the shape of the 3D array
  tvm::runtime::ShapeTuple shape({dim1, dim2, dim3});

  // Define the data type of the array elements
  DLDataType dtype{kDLFloat, 32, 1};  // float32

  // Create an empty NDArray with the specified shape and data type on CPU
  tvm::Device dev{kDLCPU, 0};
  tvm::runtime::NDArray ndarray_cpu = tvm::runtime::NDArray::Empty(shape, dtype, dev);

  // Copy data from the 1D vector to the 3D NDArray
  ndarray_cpu.CopyFromBytes(data_1d.data(), data_1d.size() * sizeof(float));

  return ndarray_cpu;
}


std::vector<float>
ReadMelData(const std::string& fname, int64_t n_rows, int64_t n_frames) {
  // Open the binary file for reading
  std::ifstream in(fname, std::ios::binary);

  if (!in)
    throw std::runtime_error("Cannot open " + fname + " for reading.");

  // Calculate the total size and reserve space
  int64_t total_size = n_rows * n_frames;
  std::vector<float> mel_data_1D(total_size);  // Allocate memory for all elements

  // Read data from the file directly into the 1D vector
  in.read(reinterpret_cast<char*>(mel_data_1D.data()), total_size * sizeof(float));

  // Check for read errors
  if (!in.good())
    throw std::runtime_error("Error occurred while reading from " + fname);

  in.close();
  return mel_data_1D;
}


Translator::Translator(const std::vector<std::string>& argv)
{
  std::string local_id = argv[0];
  std::string lib_path = argv[1];
  std::string arg_device_name = argv[2];
  auto [device_name, device_id] = DetectDevice(arg_device_name);

  try {
    TranslateModule translate(GetDevice(device_name, device_id));
    ModelPaths model = ModelPaths::Find(device_name, local_id, lib_path);
    translate_mod = mlc::llm::CreateTranslateModule(GetDevice(device_name, device_id));
    model_path = model.config.parent_path().string();
    tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(model.lib.string());
    translate_mod.GetFunction("reload")(lib, tvm::String(model_path));
  }
  catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}


std::string
Translator::generateTranscription(const std::vector<float>& input_features)
{
  // Create a 3D NDArray from the 1D vector of float
  tvm::runtime::NDArray input_features_ndarray = Create3DNDArrayFrom1DArray(input_features, 1, 80, 3000);

  int type_code = kTVMNDArrayHandle;

  // Call the TVM function with these arguments
  std::string rv = translate_mod.GetFunction("generate")(input_features_ndarray,
                                                         type_code,
                                                         tvm::String(model_path));


  return rv;

  // std::string text;

  // try {
  //   tvm::runtime::NDArray input_features_ndarray = Create3DNDArrayFrom1DArray(input_features, 1, 80, 3000);

  //   int type_code = kTVMNDArrayHandle;

  //   translate_mod.GetFunction("encode")(input_features_ndarray, type_code,
  //                                       tvm::String(model_path));
  //   translate_mod.GetFunction("decode")(tvm::String(model_path));
  //   std::string message = translate_mod.GetFunction("get_message")();
  //   text += message;

  //   bool stopped = translate_mod.GetFunction("stopped")();
  //   while (!stopped) {
  //     translate_mod.GetFunction("prefill")(tvm::String(model_path));
  //     std::string message = translate_mod.GetFunction("get_message")();
  //     text += message;
  //     stopped = translate_mod.GetFunction("stopped")();
  //   }
  // }
  // catch (const std::runtime_error& err) {
  //   std::cerr << err.what() << std::endl;
  //   return "";
  // }

  // return text;
}


std::string
Translator::generateFromFeatureVector(const std::vector<std::vector<std::vector<float>>>& data)
{
  std::vector<float> inputFeatures;

  /* Flatten the array, always O(80 * 3000) */
  for (int i = 0; i < data.size(); ++i) {
    for (int j = 0; j < data[0].size(); ++j) {
      for (int k = 0; k < data[0][0].size(); ++k)
        inputFeatures.push_back(data[i][j][k]);
    }
  }

  return generateTranscription(inputFeatures);
}


std::string
Translator::generateFromFeatureFile(const std::string& fileName)
{
  return generateTranscription(ReadMelData(fileName, 80, 3000));
}
