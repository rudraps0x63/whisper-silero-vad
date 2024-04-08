#define PICOJSON_USE_INT64
#define __STDC_FORMAT_MACROS

#include "translate.h"

#include <picojson.h>
#include <tokenizers_cpp.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/debug.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/ndarray_cache_support.h>

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>  // for std::pair
#include <vector>

#include "./metadata/model.h"
#include "./support/load_bytes_from_file.h"
#include "conversation.h"
#include "random.h"
#include "tokenizers.h"

namespace mlc {
namespace llm {

using tvm::Device;
using namespace tvm::runtime;

struct FunctionTable {
  static PackedFunc SessionFuncAsPackedFunc(Session sess, DRef sess_func, String name) {
    return PackedFunc([sess, func = std::move(sess_func), name = std::move(name)](
                          TVMArgs args, TVMRetValue* rv) -> void {
      std::vector<TVMValue> tvm_values(args.num_args + 3);
      std::vector<int> tvm_type_codes(args.num_args + 3);
      TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
      setter(0, static_cast<int>(DiscoAction::kCallPacked));
      setter(1, 0);
      setter(2, func);
      for (int i = 0; i < args.num_args; ++i) {
        tvm_values[i + 3] = args.values[i];
        tvm_type_codes[i + 3] = args.type_codes[i];
      }
      *rv = sess->CallWithPacked(
          TVMArgs(tvm_values.data(), tvm_type_codes.data(), args.num_args + 3));
    });
  }

  void Init(TVMArgValue reload_lib, Device device, int num_shards) {
    Device null_device{DLDeviceType(0), 0};
    if (num_shards > 1) {
      String lib_path{nullptr};
      try {
        lib_path = reload_lib.operator String();
      } catch (...) {
        LOG(FATAL)
            << "ValueError: In multi-GPU inference, we expect the first argument to Reload to be a "
               "string path to the model library (.so on Linux or .dll on Windows), but got: "
            << ArgTypeCode2Str(reload_lib.type_code());
      }
      constexpr const char* f_create_process_pool = "runtime.disco.create_process_pool";
      if (Registry::Get(f_create_process_pool) == nullptr) {
        LOG(FATAL) << "Cannot find process launcher `" << f_create_process_pool << "`. "
                   << "Multi-GPU inference depends on MLC LLM Python API to launch process.";
      }
      std::string ccl;
      if (device.device_type == kDLCUDA) {
        ccl = "nccl";
      } else if (device.device_type == kDLROCM) {
        ccl = "rccl";
      } else {
        LOG(FATAL) << "ValueError: Multi-GPU on device " << DLDeviceType2Str(device.device_type)
                   << " is not supported. Currently, only NCCL and RCCL are integrated.";
      }
      std::vector<int64_t> device_ids(num_shards);
      for (int i = 0; i < num_shards; ++i) {
        device_ids[i] = i;
      }
      this->use_disco = true;
      this->sess =
          Session::ProcessSession(num_shards, f_create_process_pool, "mlc_chat.cli.worker");
      this->sess->InitCCL(ccl, ShapeTuple(device_ids));
      this->disco_mod = sess->CallPacked(sess->GetGlobalFunc("runtime.disco.load_vm_module"),
                                         lib_path, null_device);
      this->mod_get_func = [this, fmodule_get_function =
                                      sess->GetGlobalFunc("runtime.ModuleGetFunction")](
                               const std::string& name) -> PackedFunc {
        DRef func = sess->CallPacked(fmodule_get_function, this->disco_mod, name, false);
        bool exists = (func->DebugGetFromRemote(0).operator PackedFunc()) != nullptr;
        if (!exists) {
          return PackedFunc(nullptr);
        }
        return SessionFuncAsPackedFunc(sess, func, name);
      };
      this->get_global_func = [this](const std::string& name) -> PackedFunc {
        return SessionFuncAsPackedFunc(sess, sess->GetGlobalFunc(name), name);
      };
      this->_InitFunctions();
      {
        Module mod = this->disco_mod->DebugGetFromRemote(0);
        this->softmax_func_ = mod->GetFunction("softmax_with_temperature");
        this->model_metadata_ = ModelMetadata::FromModule(mod);
      }
    } else {
      Module executable{nullptr};
      if (reload_lib.type_code() == kTVMModuleHandle) {
        executable = reload_lib.operator Module();
      } else {
        String lib_path = reload_lib.operator String();
        executable = tvm::runtime::Module::LoadFromFile(lib_path);
      }
      this->use_disco = false;
      auto fload_exec = executable->GetFunction("vm_load_executable");
      ICHECK(fload_exec.defined()) << "TVM runtime cannot find vm_load_executable";
      this->local_vm = fload_exec();
      this->local_vm->GetFunction("vm_initialization")(
          static_cast<int>(device.device_type), device.device_id,
          static_cast<int>(memory::AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
          static_cast<int>(memory::AllocatorType::kPooled));
      this->mod_get_func = [this](const std::string& name) -> PackedFunc {
        PackedFunc func = this->local_vm->GetFunction(name, false);
        return func;
      };
      this->get_global_func = [](const std::string& name) -> PackedFunc {
        const auto* f = tvm::runtime::Registry::Get(name);
        CHECK(f != nullptr) << "ValueError: Cannot find function " << name;
        return *f;
      };
      this->model_metadata_ = ModelMetadata::FromModule(this->local_vm);
      this->_InitFunctions();
    }
  }

  ObjectRef LoadParams(const std::string& model_path, Device device, bool use_presharded_weights) {
    if (this->use_disco) {
      DRef params{nullptr};
      if (this->model_metadata_.params.empty()) {
        std::filesystem::path fs_model_path = model_path;
        std::string metadata_path = (fs_model_path / "ndarray-cache.json").string();
        std::string ndarray_cache_metadata = LoadBytesFromFile(metadata_path);
        PackedFunc loader_create = this->get_global_func("runtime.disco.ShardLoader");

        auto load_all_func_name = use_presharded_weights
                                      ? "runtime.disco.ShardLoaderLoadAllPresharded"
                                      : "runtime.disco.ShardLoaderLoadAll";
        PackedFunc loader_load_all = this->get_global_func(load_all_func_name);
        CHECK(loader_create != nullptr);
        CHECK(loader_load_all != nullptr);
        DRef loader = loader_create(metadata_path, ndarray_cache_metadata, "", this->disco_mod);
        params = loader_load_all(loader);
      } else {
        PackedFunc loader = this->get_global_func("mlc.loader.LoadMultiGPU");
        params = loader(model_path, this->disco_mod);
      }
      return params;
    } else {
      CHECK(!use_presharded_weights) << "Use of pre-sharded weights requires more than one GPU";

      const PackedFunc* fload_cache = tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.load");
      ICHECK(fload_cache) << "TVM runtime cannot find vm.builtin.ndarray_cache.load";
      (*fload_cache)(model_path, static_cast<int32_t>(device.device_type), device.device_id);
      Array<NDArray> params;
      if (this->model_metadata_.params.empty()) {
        constexpr const char* name_loader = "vm.builtin.param_array_from_cache";
        const PackedFunc* fload_params = tvm::runtime::Registry::Get(name_loader);
        ICHECK(fload_params) << "Cannot find env function: " << name_loader;
        params = (*fload_params)("param", -1);
      } else {
        constexpr const char* name_loader = "vm.builtin.param_array_from_cache_by_name";
        const PackedFunc* fload_params = tvm::runtime::Registry::Get(name_loader);
        ICHECK(fload_params) << "Cannot find env function: " << name_loader;
        Array<String> param_names;
        param_names.reserve(this->model_metadata_.params.size());
        for (const auto& param : this->model_metadata_.params) {
          param_names.push_back(param.name);
        }
        params = (*fload_params)(param_names);
      }
      // after we get params, it is safe to simply clear the cached version
      // as these params are referenced by params_
      const PackedFunc* fclear_ndarray_cache =
          tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.clear");
      ICHECK(fclear_ndarray_cache) << "Cannot find env function vm.builtin.ndarray_cache.clear";
      (*fclear_ndarray_cache)();
      return params;
    }
  }

  void _InitFunctions() {
    this->encode_func_ = mod_get_func("encode");
    this->prefill_func_ = mod_get_func("prefill");
    this->embed_func_ = mod_get_func("embed");
    this->prefill_with_embed_func_ = mod_get_func("prefill_with_embed");
    this->decode_func_ = mod_get_func("decode");
    this->softmax_func_ = mod_get_func("softmax_with_temperature");
    this->encoding_without_cache_func_ = mod_get_func("encoding_without_cache");
    this->create_kv_cache_func_ = mod_get_func("create_kv_cache");
    if (this->create_kv_cache_func_ == nullptr) {
      this->create_kv_cache_func_ = mod_get_func("_initialize_effect");
    }
    this->reset_kv_cache_func_ = mod_get_func("reset_kv_cache");
    if (this->reset_kv_cache_func_ == nullptr) {
      this->reset_kv_cache_func_ = get_global_func("vm.builtin.attention_kv_cache_array_clear");
      support_backtracking_kv_ = true;
    } else {
      support_backtracking_kv_ = false;
    }
    this->fkvcache_array_popn_ = get_global_func("vm.builtin.attention_kv_cache_array_popn");
  }

  ObjectRef Empty(ShapeTuple shape, DataType dtype, Device device) const {
    Device null_device{DLDeviceType(0), 0};
    if (this->use_disco) {
      DRef empty_func = sess->GetGlobalFunc("runtime.disco.empty");
      return sess->CallPacked(empty_func, shape, dtype, null_device);
    } else {
      return NDArray::Empty(shape, dtype, device);
    }
  }

  ObjectRef CopyToWorker0(const NDArray& host_array) {
    Device null_device{DLDeviceType(0), 0};
    if (this->use_disco) {
      DRef array =
          Downcast<DRef>(this->Empty(host_array.Shape(), host_array.DataType(), null_device));
      sess->CopyToWorker0(host_array, array);
      return array;
    } else {
      return host_array;
    }
  }

  bool use_disco = false;
  Session sess{nullptr};
  DRef disco_mod{nullptr};
  tvm::runtime::Module local_vm{nullptr};

  TypedPackedFunc<PackedFunc(const std::string&)> mod_get_func;
  TypedPackedFunc<PackedFunc(const std::string&)> get_global_func;

  PackedFunc encode_func_;
  PackedFunc prefill_func_;
  PackedFunc embed_func_;
  PackedFunc prefill_with_embed_func_;
  PackedFunc decode_func_;
  PackedFunc encoding_without_cache_func_;
  PackedFunc softmax_func_;
  PackedFunc create_kv_cache_func_;
  PackedFunc reset_kv_cache_func_;
  bool support_backtracking_kv_;
  PackedFunc fkvcache_array_popn_;
  ModelMetadata model_metadata_;
};

//------------------------------
// Translation module
//------------------------------

class TransformerTranslateModule;

/*!
 * \brief Implements the translation module wrapper
 */
class TransformerTranslate {
  friend class TranslateModule;

 public:
  explicit TransformerTranslate(DLDevice device) : device_(device) {}

  /*!
   * \return Text describing runtime stats.
   * Todo:Make sure should we include the encode time in the prefill time
   *
   */
  std::string RuntimeStatsText() {
    std::ostringstream os;
    os << "Decode & Prefill: " << std::setprecision(1) << std::fixed
       << this->decode_total_tokens / (this->decode_total_time + this->encode_total_time)
       << " tok/s";

    return os.str();
  }

  /*!
    \brief is responsible for dynamically updating the configuration of the chat model
           based on metadata information, ensuring that the model's behavior and processing
           limits are in sync with the metadata specifications.
   */
  void UpdateConfigFromMetadata() {
    // Check for distributed inference mode
    if (ft_.use_disco) {
      return;
    }

    // Get Packed function named metadata from model
    PackedFunc fget_metadata = ft_.mod_get_func("_metadata");  // name in SLIM
    // If not found, try the old name
    if (fget_metadata == nullptr) {
      fget_metadata = ft_.mod_get_func("get_metadata");  // backward-compatible name
      if (fget_metadata == nullptr) {
        return;  // Skip if neither exists
      }
    }

    // Execute the metadata function
    ObjectRef ret = fget_metadata();
    // Get the metadata string
    std::string metadata_str = std::string(Downcast<String>(ret));
    // Parse the metadata string
    picojson::value metadata_info;
    picojson::parse(metadata_info, std::string(metadata_str));
    auto metadata = metadata_info.get<picojson::object>();

    // Check for presence of key as max_window_size or context_window_size
    std::string key = "max_window_size";
    if (!metadata.count(key)) {
      key = "context_window_size";
      ICHECK(metadata.count(key))
          << "Key \"max_window_size\" or \"context_window_size\" not found.";
    }
    ICHECK(metadata[key].is<int64_t>());
    max_window_size_ = std::min(max_window_size_, metadata[key].get<int64_t>());
  }
  /*!
   * \return Text describing verbose runtime stats.
   * Todo:Make sure should we include the encode time in the prefill time
   */
  std::string VerboseRuntimeStatsText() {
    std::ostringstream os;
    os << "------------ decode & prefill  ------------\n"
       << "throughput: " << std::setprecision(3) << std::fixed
       << this->decode_total_tokens / (this->decode_total_time + this->encode_total_time)
       << " tok/s\n"
       << "total tokens: " << this->decode_total_tokens << " tok\n"
       << "total time: " << (this->decode_total_time + this->encode_total_time) << " s\n";
    return os.str();
  }

  /*!
   * \brief Load JSON config and override options. The purpose of LoadJSONOverride()
            is to allow for dynamic changes in the model's configuration. This can be
            particularly useful in scenarios where different settings are required for
            different types of inputs or when the model is expected to adapt to changing
            conditions or requirements.
   * \param config_json A json config in picojson type that is partially specifies
   *        some of the options.
   * \param partial_update Whether it's a partial update or full update, if set to true,
   *        we perform a partial update on some of the provided options; if set to false, all
   *        options must be provided.
   * \note This function overrides existing configurations.
   */
  void LoadJSONOverride(const picojson::value& config_json, bool partial_update = false) {
    // Load config from JSON.
    picojson::object config = config_json.get<picojson::object>();
    if (config.count("temperature")) {
      CHECK(config["temperature"].is<double>());
      this->temperature_ = config["temperature"].get<double>();
    } else {
      CHECK(partial_update) << "Key \"temperature\" not found.";
    }

    if (config.count("vocab_size")) {
      CHECK(config["vocab_size"].is<int64_t>());
      this->vocab_size_ = config["vocab_size"].get<int64_t>();
    } else {
      CHECK(partial_update) << "Key \"vocab_size\" not found.";
    }

    if (config.count("bos_token_id")) {
      CHECK(config["bos_token_id"].is<int64_t>());
      int64_t bos_token_id_64 = config["bos_token_id"].get<int64_t>();

      // Check if bos_token_id_64 can be safely cast to int32_t
      if (bos_token_id_64 <= std::numeric_limits<int32_t>::max() &&
          bos_token_id_64 >= std::numeric_limits<int32_t>::min()) {
        this->bos_token_id_ = static_cast<int32_t>(bos_token_id_64);
      } else {
        LOG(FATAL) << "bos_token_id value " << bos_token_id_64
                   << " cannot be safely converted to int32_t.";
      }
    } else {
      CHECK(partial_update) << "Key \"bos_token_id\" not found.";
    }

    if (config.count("eos_token_id")) {
      CHECK(config["eos_token_id"].is<int64_t>());
      int64_t eos_token_id_64 = config["eos_token_id"].get<int64_t>();

      // Check if eos_token_id_64 can be safely cast to int32_t
      if (eos_token_id_64 <= std::numeric_limits<int32_t>::max() &&
          eos_token_id_64 >= std::numeric_limits<int32_t>::min()) {
        this->eos_token_id_ = static_cast<int32_t>(eos_token_id_64);
      } else {
        LOG(FATAL) << "eos_token_id value " << eos_token_id_64
                   << " cannot be safely converted to int32_t.";
      }
    } else {
      CHECK(partial_update) << "Key \"eos_token_id\" not found.";
    }

    if (config.count("tensor_parallel_shards")) {
      CHECK(config["tensor_parallel_shards"].is<int64_t>());
      this->num_shards_ = config["tensor_parallel_shards"].get<int64_t>();
    } else {
      this->num_shards_ = 1;
    }
    if (config.count("use_presharded_weights")) {
      CHECK(config["use_presharded_weights"].is<bool>());
      this->use_presharded_weights_ = config["use_presharded_weights"].get<bool>();
    } else {
      this->use_presharded_weights_ = false;
    }

    if (config.count("max_window_size")) {
      CHECK(config["max_window_size"].is<int64_t>());
      this->max_window_size_ =
          std::min(this->max_window_size_, config["max_window_size"].get<int64_t>());
    }
    if (config.count("context_window_size")) {
      CHECK(config["context_window_size"].is<int64_t>());
      this->max_window_size_ =
          std::min(this->max_window_size_, config["context_window_size"].get<int64_t>());
    }

    if (config.count("top_p")) {
      CHECK(config["top_p"].is<double>());
      this->top_p_ = config["top_p"].get<double>();
    } else {
      CHECK(partial_update) << "Key \"top_p\" not found.";
    }

    if (config.count("model_config")) {
      picojson::object model_config = config["model_config"].get<picojson::object>();
      if (model_config.count("max_target_positions")) {
        CHECK(model_config["max_target_positions"].is<double>());
        this->max_target_len_ = model_config["max_target_positions"].get<double>();
      }
      if (model_config.count("max_source_positions")) {
        CHECK(model_config["max_source_positions"].is<double>());
        this->max_source_len_ = model_config["max_source_positions"].get<double>();
      }
      if (model_config.count("max_length")) {
        CHECK(model_config["max_length"].is<double>());
        this->max_target_len_ = model_config["max_length"].get<double>();
      }

    } else {
      CHECK(partial_update) << "Key \"model_config\" not found.";
    }

    if (config.count("suppress_tokens")) {
      // Check if it's not null and is an array
      if (!config["suppress_tokens"].is<picojson::null>() &&
          config["suppress_tokens"].is<picojson::array>()) {
        this->suppress_tokens_.clear();
        for (const picojson::value& v : config["suppress_tokens"].get<picojson::array>()) {
          CHECK(v.is<std::int64_t>());
          this->suppress_tokens_.push_back(v.get<std::int64_t>());
        }
      }
    }

    if (config.count("begin_suppress_tokens")) {
      if (!config["begin_suppress_tokens"].is<picojson::null>() &&
          config["begin_suppress_tokens"].is<picojson::array>()) {
        this->begin_suppress_tokens_.clear();
        for (const picojson::value& v : config["begin_suppress_tokens"].get<picojson::array>()) {
          CHECK(v.is<std::int64_t>());
          this->begin_suppress_tokens_.push_back(v.get<std::int64_t>());
        }
      }
    }

    if (config.count("forced_decoder_ids")) {
      if (!config["forced_decoder_ids"].is<picojson::null>() &&
          config["forced_decoder_ids"].is<picojson::array>()) {
        this->forced_tokens_.clear();
        for (const picojson::value& v : config["forced_decoder_ids"].get<picojson::array>()) {
          CHECK(v.is<picojson::array>());
          const picojson::array& arr = v.get<picojson::array>();
          CHECK(arr.size() == 2);
          int first = arr[0].get<int64_t>();
          std::optional<int> second;
          if (!arr[1].is<picojson::null>()) {
            second = arr[1].get<int64_t>();
          }
          this->forced_tokens_.push_back(std::make_pair(first, second));
        }
        // Process the forced tokens and store them in a map
        forced_decoder_map_ = ProcessForcedTokens();
      }
    }
  }

  /*!
   * \brief Load JSON config and override options.
   * \param config_str A json config string that partially specifies some of the options.
   * \param partial_update Whether it's a partial update or full update, if set to true,
   *        we perform a partial update on some of the provided options; if set to false, all
   *        options must be provided.
   * \note This function overrides existing configurations.
   */
  void LoadJSONOverride(const std::string& config_str, bool partial_update = false) {
    picojson::value config_json;
    std::string err = picojson::parse(config_json, config_str);
    if (!err.empty()) {
      LOG(FATAL) << err;
      return;
    }
    LoadJSONOverride(config_json, partial_update);
  }

  std::string GetConfigJSON() const { return SerializeConfigToJSONValue().serialize(true); }

  /*!
   * \brief Reload model, tokenizers and configurations from the specified model path.
   * \param reload_lib The module to reload, it can either be a path to the library or a tvm Module.
   * \param model_path The path to search for models.
   * \param app_config_json The JSON string used to partially override the configuration loaded from
   * disk, default to empty string.
   */
  void Reload(TVMArgValue reload_lib, String model_path, String app_config_json = "") {
    // Step 1. Process config json string.
    {
      std::ifstream config_istream((model_path + "/mlc-chat-config.json").c_str());
      std::ostringstream config_ostream;
      ICHECK(config_istream);
      config_ostream << config_istream.rdbuf();
      std::string config_str = config_ostream.str();
      LoadJSONOverride(config_str, false);
      if (!app_config_json.empty()) {
        // Override configuration from app_config_json.
        LoadJSONOverride(app_config_json, true);
      }
    }
    // Step 2. Set tokenizer.
    this->tokenizer_ = TokenizerFromPath(model_path);

    // Step 3. Initialize vm, we use the packed function mechanism
    // so there is no explicit abi dependency on these extra
    // classes other than basic tvm runtime.
    this->ft_.Init(reload_lib, device_, this->num_shards_);
    UpdateConfigFromMetadata();
    if (this->sliding_window_size_ == -1) {
      CHECK(max_window_size_ != std::numeric_limits<int64_t>::max())
          << "Key \"max_window_size\" not found.";
    }
    // Step 4. Initialize sample functions.
    auto fsample_topp_from_prob_ptr =
        tvm::runtime::Registry::Get("vm.builtin.sample_top_p_from_prob");
    ICHECK(fsample_topp_from_prob_ptr)
        << "Cannot find env function vm.builtin.sample_top_p_from_prob";
    fsample_topp_from_prob_ = *fsample_topp_from_prob_ptr;
    auto fsample_topp_from_logits_ptr =
        tvm::runtime::Registry::Get("vm.builtin.sample_top_p_from_logits");
    ICHECK(fsample_topp_from_logits_ptr)
        << "Cannot find env function vm.builtin.sample_top_p_from_logits";
    fsample_topp_from_logits_ = *fsample_topp_from_logits_ptr;
    // Step 5. Load params in nd-array cache.
    this->params_ = ft_.LoadParams(model_path, device_, use_presharded_weights_);
    // Step 6. KV cache creation.
    this->kv_cache_ = ft_.create_kv_cache_func_();
    // Step 7. Pre-allocate fixed size ndarray
    this->temperature_arr_ = NDArray::Empty({}, DataType::Float(32), device_);
    float temperature = static_cast<float>(this->temperature_);
    this->temperature_arr_.CopyFromBytes(&temperature, sizeof(float));
    if (ft_.use_disco) {
      Device null_device{DLDeviceType(0), 0};
      this->input_tokens_decode_ =
          Downcast<DRef>(ft_.Empty(ShapeTuple({1, 1}), DataType::Int(32), null_device));
    }
  }

  /*! \brief reset the runtime stats. */
  void ResetRuntimeStats() {
    this->prefill_total_tokens = 0;
    this->decode_total_tokens = 0;
    this->encode_total_time = 0;
    this->prefill_total_time = 0;
    this->decode_total_time = 0;
    this->sample_total_time = 0;
  }

  // get statically allocated input token
  // NDArray GetInputTokenNDArray(const std::vector<int32_t>& token_ids) {
  //   // try realloc
  //   if (!input_token_ids_.defined()) {
  //     int64_t init_size = 2048;
  //     while (init_size < static_cast<int64_t>(token_ids.size())) {
  //       init_size *= 2;
  //     }
  //     input_token_ids_ = NDArray::Empty({1, init_size}, DataType::Int(32), device_);
  //   } else {
  //     int64_t init_size = input_token_ids_->shape[1];
  //     while (init_size < static_cast<int64_t>(token_ids.size())) {
  //       init_size *= 2;
  //     }
  //     if (init_size != input_token_ids_->shape[1]) {
  //       input_token_ids_ = NDArray::Empty({1, init_size}, DataType::Int(32), device_);
  //     }
  //   }
  //   ICHECK_LE(token_ids.size(), input_token_ids_->shape[1]) << "Input tokens exceed window size";
  //   NDArray view = input_token_ids_.CreateView(
  //       ShapeTuple({1, static_cast<int64_t>(token_ids.size())}), input_token_ids_->dtype);
  //   if (token_ids.size() > 0) {
  //     view.CopyFromBytes(token_ids.data(), token_ids.size() * sizeof(int32_t));
  //   }
  //   return view;
  // }
  NDArray GetInputTokenNDArray(const std::vector<int32_t>& token_ids,
                               const std::string& encode_or_decode = "decode") {
    NDArray* token_ids_ptr = nullptr;

    if (encode_or_decode == "decode") {
      // Use output_token_ids_ for decoding
      token_ids_ptr = &output_token_ids_;
    } else {
      // Use input_token_ids_ for encoding
      token_ids_ptr = &input_token_ids_;
    }

    // Try realloc
    if (!token_ids_ptr->defined()) {
      int64_t init_size = 2048;
      while (init_size < static_cast<int64_t>(token_ids.size())) {
        init_size *= 2;
      }
      *token_ids_ptr = NDArray::Empty({1, init_size}, DataType::Int(32), device_);
    } else {
      int64_t init_size = token_ids_ptr->Shape()[1];
      while (init_size < static_cast<int64_t>(token_ids.size())) {
        init_size *= 2;
      }
      if (init_size != token_ids_ptr->Shape()[1]) {
        *token_ids_ptr = NDArray::Empty({1, init_size}, DataType::Int(32), device_);
      }
    }
    ICHECK_LE(token_ids.size(), token_ids_ptr->Shape()[1]) << "Input tokens exceed window size";
    NDArray view = token_ids_ptr->CreateView(
        ShapeTuple({1, static_cast<int64_t>(token_ids.size())}), token_ids_ptr->DataType());
    if (token_ids.size() > 0) {
      view.CopyFromBytes(token_ids.data(), token_ids.size() * sizeof(int32_t));
    }
    return view;
  }

  /*

  NDArray GetInputTokenNDArray(const std::vector<std::vector<std::vector<float>>>& input_features) {
    // Calculate the size of each dimension
    int64_t dim1 = input_features.size();
    int64_t dim2 = input_features.empty() ? 0 : input_features[0].size();
    int64_t dim3 =
        input_features.empty() || input_features[0].empty() ? 0 : input_features[0][0].size();

    // Create the NDArray
    if (!input_token_ids_.defined() || input_token_ids_->shape[0] < dim1 ||
        input_token_ids_->shape[1] < dim2 || input_token_ids_->shape[2] < dim3) {
      input_token_ids_ = NDArray::Empty({dim1, dim2, dim3}, DataType::Float(32), device_);
    }
    // Flatten the 3D vector into a 1D vector
    std::vector<float> flat_input_features;
    for (const auto& matrix : input_features) {
      for (const auto& row : matrix) {
        for (const auto& element : row) {
          flat_input_features.push_back(element);
        }
      }
    }

    // Copy the data into the NDArray
    input_token_ids_.CopyFromBytes(flat_input_features.data(),
                                   flat_input_features.size() * sizeof(float));



    return input_token_ids_;
  }



  // Function to create a 3D NDArray from a 1D vector of float
  tvm::runtime::NDArray Create3DNDArrayFrom1DArray(const std::vector<float>& data_1d, int dim1,
                                                   int dim2, int dim3) {
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

    // Create an empty NDArray on Vulkan with the same shape and data type.
    NDArray ndarray_vulkan = NDArray::Empty(ndarray_cpu.Shape(), ndarray_cpu.DataType(), device_);

    // Copy the data to Vulkan.
    ndarray_cpu.CopyTo(ndarray_vulkan);

    // Synchronize to ensure the copy is completed (if needed).
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);

    return ndarray_vulkan;
  }


  NDArray GetInputFeatureNDArray(const std::vector<float>& data_1d, int dim1, int dim2, int dim3) {
    // Create an empty NDArray on Vulkan with the same shape and data type.
    NDArray ndarray_vulkan = NDArray::Empty({dim1, dim2, dim3}, DataType::Float(32), device_);

    // Copy the data to Vulkan.
    ndarray_vulkan.CopyFromBytes(data_1d.data(), data_1d.size() * sizeof(float));

    return ndarray_vulkan;
  }
  */

  /*
  * \brief Given the text input, generate the embedding of the tokenized input.
           param inp The input text string.
  * \return the embedding of the tokenized input.
  */

  void EncodeStep(std::string input, int type_code, String generation_config_str = "") {
    this->ResetKVCache();
    this->ResetRuntimeStats();

    CHECK(ft_.encode_func_.defined()) << "Make sure to build the function with the encode";

    // process generation settings
    picojson::object generation_config =
        this->LoadGenerationConfigFromString(generation_config_str);

    std::vector<int32_t> tokens = tokenizer_->Encode(input);

    int64_t token_len = static_cast<int64_t>(tokens.size());

    if (token_len == 0) {
      embedding_ = NDArray::Empty({}, DataType::Float(32), device_);
    } else {
      CHECK(ft_.encode_func_.defined()) << "Make sure to build the function with the encode";

      auto tstart = std::chrono::high_resolution_clock::now();

      NDArray input_data = this->GetInputTokenNDArray(tokens, "encode");

      ObjectRef embedding = ft_.encode_func_(ft_.CopyToWorker0(input_data), kv_cache_, params_);

      input_seq_len_ = token_len;

      auto tend = std::chrono::high_resolution_clock::now();

      NDArray embedding_ndarray = extractNDArrayEncodeStep(embedding);

      this->encode_total_time += static_cast<double>((tend - tstart).count()) / 1e9;

      embedding_ = embedding_ndarray;

      std::cout << "[" << embedding_.Shape()[0] << "," << embedding_.Shape()[1] << ","
                << embedding_.Shape()[2] << "]" << std::endl;
      std::cout << input << std::endl;
      for (auto token : tokens) {
        std::cout << token << " ";
      }
    }
  }

  NDArray extractNDArrayEncodeStep(const tvm::runtime::ObjectRef& obj_ref) {
    tvm::runtime::NDArray ndarray;
    // Try to cast to Optional<ObjectRef>
    auto opt_obj_ref = obj_ref.as<tvm::runtime::ObjectRef>();
    if (opt_obj_ref.defined()) {
      // It's an Optional<ObjectRef>, now check the contained value
      const tvm::runtime::ObjectRef& contained_obj_ref = opt_obj_ref.value();

      // Check if the contained value is an NDArray
      auto opt_ndarray = contained_obj_ref.as<tvm::runtime::NDArray>();
      if (opt_ndarray.defined()) {
        // The contained value is an NDArray
        // std::cout << "Object is an Optional containing an NDArray" << std::endl;
        // get the NDarray
        ndarray = opt_ndarray.value();
      }
      // Check if the contained value is an Array<NDArray>
      else {
        auto opt_array = contained_obj_ref.as<tvm::runtime::Array<tvm::runtime::NDArray>>();
        if (opt_array.defined()) {
          // The contained value is an Array<NDArray>
          // std::cout << "Object is an Optional containing an Array<NDArray>" << std::endl;
          tvm::runtime::Array<tvm::runtime::NDArray> array = opt_array.value();
          // Add code to inspect each NDArray in the array
          // std::cout << "Number of elements in array: " << array.size() << std::endl;
          // Extract the non zero element of the array and store it into ndarray
          for (const auto& item : array) {
            if (item.Shape().size() != 0 && std::all_of(item.Shape().begin(), item.Shape().end(),
                                                        [](int64_t i) { return i != 0; })) {
              ndarray = item;
              break;
            }
          }
        }
      }
    }
    return ndarray;
  }

  /*
  * \brief Given the input_features, generate the embedding of the tokenized input.
        param inp The input text string.
  * \return the embedding of the tokenized input.
  */
  void EncodeStep(NDArray ndarray, int type_code, String generation_config_str = "") {
    // Reset the KV cache and runtime stats
    this->ResetKVCache();
    this->ResetRuntimeStats();

    CHECK(ft_.encode_func_.defined()) << "Make sure to build the function with the encode";

    // Extract arguments based on their type code
    NDArray input_data_device;
    String input_text_cpu;

    ICHECK(type_code == kTVMNDArrayHandle) << "Input data must be a NDArray.";

    if (type_code == kTVMNDArrayHandle) {
      input_data_device = ndarray.CopyTo(device_);
    }

    // Ensure input_data is not null
    // CHECK(input_data_cpu != nullptr) << "Input data NDArray is null.";

    // get the device type and device id of input_data

    // Check the shape of the input data
    CHECK(input_data_device.Shape()[0] == 1) << "Batch size must be 1";
    CHECK(input_data_device.Shape()[1] == 80) << "Number of rows must be 80";
    CHECK(input_data_device.Shape()[2] == 3000) << "Number of frames must be 3000";

    CHECK(params_.defined()) << "Make sure to load the params before calling encode";

    auto tstart = std::chrono::high_resolution_clock::now();

    ObjectRef embedding =
        ft_.encode_func_(ft_.CopyToWorker0(input_data_device), kv_cache_, params_);

    auto tend = std::chrono::high_resolution_clock::now();

    NDArray embedding_ndarray = extractNDArrayEncodeStep(embedding);

    this->encode_total_time += static_cast<double>((tend - tstart).count()) / 1e9;

    // dont need this logic. Can be removed latter
    /*
    if (embedding_.defined() && embedding_->data != nullptr) {
      // embedding_ contains dataEncodeStep
      std::cout << "Embedding contains data." << std::endl;
      std::cout << "Embedding shape: [" << embedding_.Shape()[0] << ", " << embedding_.Shape()[1]
                << "]" << std::endl;
    } else {
      // embedding_ does not contain data or is not properly initialized
      // std::cout << "Embedding does not contain data or is not initialized." << std::endl;
      embedding_ = embedding_ndarray;
    }
    */
    embedding_ = embedding_ndarray;
  }

  tvm::runtime::Array<tvm::runtime::Array<tvm::runtime::NDArray>> convertToNestedArray(
      const tvm::runtime::Array<tvm::runtime::ObjectRef>& outer_array) {
    tvm::runtime::Array<tvm::runtime::Array<tvm::runtime::NDArray>> nested_array;
    for (const auto& element : outer_array) {
      try {
        // Cast each element of the outer array to Array<NDArray>
        tvm::runtime::Array<tvm::runtime::NDArray> inner_array =
            tvm::runtime::Downcast<tvm::runtime::Array<tvm::runtime::NDArray>>(element);

        // Add the inner array to the nested array
        nested_array.push_back(inner_array);
      } catch (const tvm::runtime::Error& e) {
        std::cerr << "Error casting element to Array<NDArray>: " << e.what() << std::endl;
      }
    }

    return nested_array;
  }

  tvm::runtime::NDArray extractLogitEncoderKeyValue(tvm::runtime::ObjectRef& obj) {
    tvm::runtime::NDArray logits_on_device;
    tvm::runtime::Array<tvm::runtime::ObjectRef> all_encoder_key_value_outer_array;
    tvm::runtime::Array<tvm::runtime::Array<tvm::runtime::NDArray>> all_encoder_key_value;

    try {
      // Cast the ObjectRef to an Array of ObjectRef
      auto outer_array = tvm::runtime::Downcast<tvm::runtime::Array<tvm::runtime::ObjectRef>>(obj);

      // Ensure the outer array has at least one element and cast it
      if (outer_array.size() > 0) {
        auto inner_array =
            tvm::runtime::Downcast<tvm::runtime::Array<tvm::runtime::ObjectRef>>(outer_array[0]);

        // Ensure the inner array has the expected elements and cast them
        if (inner_array.size() > 1) {
          logits_on_device = tvm::runtime::Downcast<tvm::runtime::NDArray>(inner_array[0]);
          all_encoder_key_value_outer_array =
              tvm::runtime::Downcast<tvm::runtime::Array<tvm::runtime::ObjectRef>>(inner_array[1]);
          // Convert all_encoder_key_value_outer_array to Array<Array<NDArray>>
          all_encoder_key_value = convertToNestedArray(all_encoder_key_value_outer_array);

        } else {
          std::cerr << "Inner array does not contain the expected number of elements." << std::endl;
        }
        // get the kv_cache
        // kv_cache_ =
        //    tvm::runtime::Downcast<tvm::runtime::Array<tvm::runtime::ObjectRef>>(outer_array[1]);

      } else {
        std::cerr << "Outer array is empty." << std::endl;
      }
    } catch (const tvm::runtime::Error& e) {
      // Handle the error: the cast failed
      std::cerr << "Cast failed: " << e.what() << std::endl;
    }

    all_encoder_key_value_ = all_encoder_key_value;
    return logits_on_device;
  }

  NDArray DecodeStep() {
    // if ouptut_ids_ is not empty, then clear the output_ids_
    if (!output_ids_.empty()) {
      output_ids_.clear();
    }

    ICHECK(output_ids_.empty());

    // Add the bos_token_id to the output_ids_
    output_ids_.push_back(bos_token_id_);

    int32_t last_token = output_ids_.back();

    // Check the decode function is defined
    CHECK(ft_.decode_func_.defined()) << "Make sure to build the function with the decode";

    ObjectRef input_data;

    if (ft_.use_disco) {
      ft_.sess->CopyToWorker0(this->GetInputTokenNDArray({last_token}, "decode"),
                              input_tokens_decode_);
      input_data = input_tokens_decode_;
    } else {
      input_data = ft_.CopyToWorker0(this->GetInputTokenNDArray({last_token}, "decode"));
    }

    // Get the logits of the next token
    ObjectRef ret = ft_.decode_func_(Downcast<NDArray>(input_data),
                                     ShapeTuple({1}),  // generated_tokens_length_shape
                                     embedding_, kv_cache_, params_);

    // get the logits
    NDArray logits_on_device = extractLogitEncoderKeyValue(ret);

    return logits_on_device;
  }

  void DecodeStep(String generation_config_str) {
    picojson::object generation_config =
        this->LoadGenerationConfigFromString(generation_config_str);

    // if ouptut_ids_ is not empty, then clear the output_ids_
    if (!output_ids_.empty()) {
      output_ids_.clear();
    }

    ICHECK(output_ids_.empty());

    // Add the bos_token_id to the output_ids_
    output_ids_.push_back(bos_token_id_);

    int32_t last_token = output_ids_.back();

    auto tstart = std::chrono::high_resolution_clock::now();

    // Check the decode function is defined
    CHECK(ft_.decode_func_.defined()) << "Make sure to build the function with the decode";

    ObjectRef input_data;

    if (ft_.use_disco) {
      ft_.sess->CopyToWorker0(this->GetInputTokenNDArray({last_token}, "decode"),
                              input_tokens_decode_);
      input_data = input_tokens_decode_;
    } else {
      input_data = ft_.CopyToWorker0(this->GetInputTokenNDArray({last_token}, "decode"));
    }

    // shape of input_data
    // type cast to NDArray
    // std::cout << "diagonistics" << std::endl;
    // std::cout << output_token_ids_.Shape()[0] << ", " << output_token_ids_.Shape()[1] <<
    // std::endl; std::cout << tvm::runtime::Downcast<NDArray>(input_data).Shape()[0] << " "
    //           << tvm::runtime::Downcast<NDArray>(input_data).Shape()[1] << std::endl;
    // // std::cout << input_data.Shape()[0] << " " << input_data.Shape()[1] << std::endl;
    // std::cout << last_token << std::endl;
    // std::cout << __func__ << ':' << __LINE__ << std::endl;

    // std::cout << embedding_.Shape()[0] << ", " << embedding_.Shape()[1] << ", "
    //           << embedding_.Shape()[2] << std::endl;

    // Get the logits of the next token
    ObjectRef ret = ft_.decode_func_(Downcast<NDArray>(input_data),
                                     ShapeTuple({1}),  // generated_tokens_length_shape
                                     embedding_, kv_cache_, params_);

    std::cout << __func__ << ':' << __LINE__ << '\n';

    // get the logits
    NDArray logits_on_device = extractLogitEncoderKeyValue(ret);

    // update the logits on cpu
    // UpdateLogitsOrProbOnCPUSync(logits_on_device);

    // get the next token
    int32_t next_token = this->SampleTokenFromLogits(logits_on_device, generation_config);

    auto tend = std::chrono::high_resolution_clock::now();

    this->decode_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    this->decode_total_tokens += 1;

    // Update the output_ids_ with the next token
    output_ids_.push_back(next_token);

    // clear the output_message if not empty else push the first token to it.
    if (!output_message_.empty()) {
      output_message_.clear();
    } else {
      output_message_ = tokenizer_->Decode(output_ids_);  // Decode the token
    }
  }

  void PrefillStep(String generation_config_str) {
    picojson::object generation_config =
        this->LoadGenerationConfigFromString(generation_config_str);

    ICHECK(!output_ids_.empty());

    int32_t token = output_ids_.back();

    ObjectRef input_data;

    if (ft_.use_disco) {
      ft_.sess->CopyToWorker0(this->GetInputTokenNDArray(output_ids_, "decode"),
                              input_tokens_decode_);
      input_data = input_tokens_decode_;
    } else {
      input_data = ft_.CopyToWorker0(this->GetInputTokenNDArray({token}, "decode"));
    }

    // Check the prefill function is defined
    CHECK(ft_.prefill_func_.defined()) << "Make sure to build the function with the prefill";

    auto tstart = std::chrono::high_resolution_clock::now();

    int64_t generated_tokens_length = static_cast<int64_t>(output_ids_.size());

    ShapeTuple cur_pos_shape = ShapeTuple({generated_tokens_length});

    // Get the logits of the next token
    ObjectRef ret = ft_.prefill_func_(input_data, cur_pos_shape,  // generated_tokens_length_shape
                                      all_encoder_key_value_, kv_cache_, params_);

    // get the outer array
    auto outer_array = Downcast<tvm::runtime::Array<tvm::runtime::ObjectRef>>(ret);

    // logits on device
    NDArray logits_on_device = Downcast<NDArray>(outer_array[0]);

    // Only copy logits to the host if necessary operations are to be performed
    if (!suppress_tokens_.empty() || !begin_suppress_tokens_.empty() ||
        forced_decoder_map_.count(decode_total_tokens) > 0) {
      ProcessLogits(forced_decoder_map_, logits_on_device);
    }

    // update the logits on cpu
    // UpdateLogitsOrProbOnCPUSync(logits_on_device);

    // get the next token
    int32_t next_token = this->SampleTokenFromLogits(logits_on_device, generation_config);

    auto tend = std::chrono::high_resolution_clock::now();

    this->decode_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    this->decode_total_tokens += 1;

    // Update the output_ids_ with the next token
    output_ids_.push_back(next_token);

    std::vector<int32_t> tokens = {next_token};
    output_message_ = tokenizer_->Decode(tokens);  // Decode the token
  }

  NDArray PrefillStep() {
    ICHECK(!output_ids_.empty());

    int32_t token = output_ids_.back();

    ObjectRef input_data;

    if (ft_.use_disco) {
      ft_.sess->CopyToWorker0(this->GetInputTokenNDArray(output_ids_, "decode"),
                              input_tokens_decode_);
      input_data = input_tokens_decode_;
    } else {
      input_data = ft_.CopyToWorker0(this->GetInputTokenNDArray({token}, "decode"));
    }

    // Check the prefill function is defined
    CHECK(ft_.prefill_func_.defined()) << "Make sure to build the function with the prefill";

    auto tstart = std::chrono::high_resolution_clock::now();

    int64_t generated_tokens_length = static_cast<int64_t>(output_ids_.size());

    ShapeTuple cur_pos_shape = ShapeTuple({generated_tokens_length});

    // Get the logits of the next token
    ObjectRef ret = ft_.prefill_func_(input_data, cur_pos_shape,  // generated_tokens_length_shape
                                      all_encoder_key_value_, kv_cache_, params_);

    // get the outer array
    auto outer_array = Downcast<tvm::runtime::Array<tvm::runtime::ObjectRef>>(ret);

    // logits on device
    NDArray logits_on_device = Downcast<NDArray>(outer_array[0]);

    return logits_on_device;
  }

  void SuppressColumns(tvm::runtime::NDArray& arr, const std::vector<int64_t>& columns) {
    auto shape = arr.Shape();
    int64_t depth = shape[0];
    int64_t rows = shape[1];
    int64_t cols = shape[2];
    float* data = static_cast<float*>(arr->data);

    for (int64_t batch = 0; batch < depth; ++batch) {
      for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col : columns) {
          if (col >= cols) {
            continue;  // Skip if the column index is out of bounds
          }
          int64_t idx = (batch * rows * cols) + (row * cols) + col;
          data[idx] = -std::numeric_limits<float>::infinity();
        }
      }
    }
  }

  void SetScoresAndToken(tvm::runtime::NDArray& arr, int64_t token) {
    auto shape = arr.Shape();
    int64_t depth = shape[0];
    int64_t rows = shape[1];
    int64_t cols = shape[2];
    float* data = static_cast<float*>(arr->data);

    for (int64_t batch = 0; batch < depth; ++batch) {
      for (int64_t row = 0; row < rows; ++row) {
        // Set all scores in the current row to negative infinity
        std::fill_n(&data[(batch * rows * cols) + (row * cols)], cols,
                    -std::numeric_limits<float>::infinity());

        // Check if the token index is within bounds
        if (token < cols) {
          //******DEBUG********
          // std::cout << "token is " << token << __func__ << ':' << __LINE__ << std::endl;
          //******DEBUG********
          // Set the score for the current token to 0.0f
          data[(batch * rows * cols) + (row * cols) + token] = 0.0f;
        }
      }
    }
  }

  std::unordered_map<int64_t, int64_t> ProcessForcedTokens() {
    std::unordered_map<int64_t, int64_t> forced_decoder_map;
    for (const auto& token_pair : forced_tokens_) {
      // Check if the optional has a value before inserting
      if (token_pair.second.has_value()) {
        // Cast to int64_t to match the type of forced_decoder_map
        forced_decoder_map[static_cast<int64_t>(token_pair.first)] =
            static_cast<int64_t>(token_pair.second.value());
      }
    }
    return forced_decoder_map;
  }

  void ProcessLogits(std::unordered_map<int64_t, int64_t>& forced_decoder_map,
                     NDArray& logits_on_device) {
    UpdateLogitsOrProbOnCPUSync(logits_on_device);

    // Suppress the columns of the logits
    if (!suppress_tokens_.empty()) {
      SuppressColumns(logits_on_cpu_, suppress_tokens_);
    }

    // Suppress the columns of the logits
    std::vector<int32_t>& generated_tokens = output_ids_;
    if (!begin_suppress_tokens_.empty()) {
      if (generated_tokens.size() == 1 + forced_tokens_.back().first) {
        SuppressColumns(logits_on_cpu_, begin_suppress_tokens_);
      }
    }

    // Force the token at the current position
    if (!forced_decoder_map.empty()) {
      int64_t generation_idx = generated_tokens.size();
      auto current_token = forced_decoder_map.find(generation_idx);
      if (current_token != forced_decoder_map.end()) {
        //******DEBUG********
        // std::cout << generation_idx << " " << current_token->first << " " << current_token->second << " " << __func__ << ':' << __LINE__ << std::endl;
        //******DEBUG********
        SetScoresAndToken(logits_on_cpu_, current_token->second);
      }
    }
  }

  bool CheckStopCondition() {
    ICHECK(!output_ids_.empty());
    std::vector<int32_t>& generated_tokens = output_ids_;
    if (generated_tokens.back() == eos_token_id_ || generated_tokens.size() >= max_target_len_) {
      return true;
    }
    return false;
  }

  bool ProcessNextToken(std::unordered_map<int64_t, int64_t>& forced_decoder_map,
                        NDArray& logits_on_device,
                        std::chrono::high_resolution_clock::time_point& tstart,
                        picojson::object generation_config) {
    std::vector<int32_t>& generated_tokens = output_ids_;

    // Only copy logits to the host if necessary operations are to be performed
    if (!suppress_tokens_.empty() || !begin_suppress_tokens_.empty() ||
        forced_decoder_map.count(generated_tokens.size()) > 0) {
      ProcessLogits(forced_decoder_map, logits_on_device);

      //******DEBUG********
      // print the index of the token which is not -inf
      // float* data = static_cast<float*>(logits_on_cpu_->data);
      // for (int i = 0; i < logits_on_cpu_.Shape()[2]; i++) {
      //   if (data[0 * 80 * i + 0 * 50264 + i] != -std::numeric_limits<float>::infinity()) {
      //     std::cout << "logits_on_cpu_ for token " << i << " "
      //               << "is 0 and rest is -inf in " << __func__ << " " << __LINE__ << std::endl;
      //   }
      // }
      // std::cout << "logits_on_cpu_ for token " << 50259 << " " << data[50259] << " " << __func__
      //           << " " << __LINE__ << std::endl;
      // std::cout << "logits_on_cpu_ for token " << 50359 << " " << data[50359] << " " << __func__
      //           << " " << __LINE__ << std::endl;
      // std::cout << "logits_on_cpu_ for token " << 50363 << " " << data[50363] << " " << __func__
      //           << " " << __LINE__ << std::endl;

      //******DEBUG********

      // get the next token
      int32_t next_token = this->SampleTokenFromLogits(logits_on_device, generation_config);

      // std::cout << "next_token: " << next_token << " " << tokenizer_->Decode({next_token}) << std::endl;

      // Update the output_ids_ with the next token
      output_ids_.push_back(next_token);

      // Update the total decode time and tokens outside the if condition
      auto tend = std::chrono::high_resolution_clock::now();
      this->decode_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
      this->decode_total_tokens += 1;

      // Reset the start time for the next iteration
      tstart = std::chrono::high_resolution_clock::now();

      if (CheckStopCondition()) {
        return false;
      } else {
        // Get the logits of the next token
        logits_on_device = PrefillStep();
      }
    }
    return true;
  }

  void Generate(NDArray ndarray, int type_code, String generation_config_str = "") {
    picojson::object generation_config =
        this->LoadGenerationConfigFromString(generation_config_str);

    // Process the forced tokens
    std::unordered_map<int64_t, int64_t> forced_decoder_map = ProcessForcedTokens();

    EncodeStep(ndarray, type_code, generation_config_str);

    // Start timing before the loop
    auto tstart = std::chrono::high_resolution_clock::now();

    // Move the declaration of logits_on_device outside the loop to avoid re-declaration on each
    // iteration
    NDArray logits_on_device = DecodeStep();

    // Run the loop until a condition is met to end it.
    while (ProcessNextToken(forced_decoder_map, logits_on_device, tstart, generation_config)) {
      // Loop until ProcessNextToken returns false, indicating the stop condition has been met
    }

    // check if ouput_message_ is empty and if not, clear the contents
    if (!output_message_.empty()) {
      output_message_.clear();
    }

    output_message_ = tokenizer_->Decode({output_ids_.begin(), output_ids_.end()});
  }

  std::string GetOutputMessage() { return output_message_; }

  NDArray GetEmbeddingNDArray() { return embedding_; }

 private:
  picojson::value SerializeConfigToJSONValue() const {
    picojson::object config;
    config["temperature"] = picojson::value(this->temperature_);
    config["top_p"] = picojson::value(this->top_p_);
    return picojson::value(config);
  }

  picojson::object LoadGenerationConfigFromString(const std::string& generation_config_str) {
    std::ifstream config_istream((generation_config_str + "/mlc-chat-config.json").c_str());
    std::ostringstream config_ostream;
    ICHECK(config_istream);
    config_ostream << config_istream.rdbuf();
    std::string config_str = config_ostream.str();

    picojson::object generation_config = picojson::object();
    if (!generation_config_str.empty()) {
      picojson::value generation_config_json;
      std::string err = picojson::parse(generation_config_json, config_str);
      if (!err.empty()) {
        std::cerr << "JSON parsing error: " << err << std::endl;
      }
      if (generation_config_json.is<picojson::object>()) {
        generation_config = generation_config_json.get<picojson::object>();
      } else {
        std::cerr << "Input string is not a JSON object" << std::endl;
      }
    }
    return generation_config;
  }

  void ReadGenerationConfig(picojson::object generation_config, double* gen_temperature,
                            NDArray* gen_temperature_arr, double* gen_repetition_penalty,
                            double* gen_presence_penalty, double* gen_frequency_penalty,
                            double* gen_top_p) {
    if (generation_config.count("temperature")) {
      CHECK(generation_config["temperature"].is<double>());
      *gen_temperature = generation_config["temperature"].get<double>();

      *gen_temperature_arr = NDArray::Empty({}, DataType::Float(32), device_);
      float temperature_cast = static_cast<float>(*gen_temperature);
      gen_temperature_arr->CopyFromBytes(&temperature_cast, sizeof(float));
    } else {
      *gen_temperature = this->temperature_;
      *gen_temperature_arr = this->temperature_arr_;
    }
    if (generation_config.count("top_p")) {
      CHECK(generation_config["top_p"].is<double>());
      *gen_top_p = generation_config["top_p"].get<double>();
    } else {
      *gen_top_p = this->top_p_;
    }
  }

  // /*!
  //  * \brief Sample output token from logits on device
  //  */
  // int32_t SampleTokenFromLogits(NDArray logits_on_device,
  //                               picojson::object generation_config = picojson::object()) {
  //   // prepare generation settings
  //   // the generation_config will not override the original config
  //   // since is only used for this generation
  //   double gen_temperature;
  //   double gen_repetition_penalty;
  //   double gen_presence_penalty;
  //   double gen_frequency_penalty;
  //   double gen_top_p;
  //   this->ReadGenerationConfig(generation_config, &gen_temperature, &this->temperature_arr_,
  //                              &gen_repetition_penalty, &gen_presence_penalty,
  //                              &gen_frequency_penalty, &gen_top_p);

  //   // update logits
  //   if (gen_presence_penalty != 0.0f || gen_frequency_penalty != 0.0f) {
  //     this->UpdateLogitsOrProbOnCPUSync(logits_on_device);
  //     if (gen_temperature >= 1e-6f) {
  //       this->ApplySoftmaxWithTemperatureOnCPU(gen_temperature);
  //     }
  //   } else if (gen_repetition_penalty != 1.0f) {
  //     this->UpdateLogitsOrProbOnCPUSync(logits_on_device);
  //     if (gen_temperature >= 1e-6f) {
  //       this->ApplySoftmaxWithTemperatureOnCPU(gen_temperature);
  //     }
  //   } else {
  //     if (gen_temperature < 1e-6f) {
  //       this->UpdateLogitsOrProbOnCPUSync(logits_on_device);
  //     } else {
  //       this->UpdateLogitsOrProbOnCPUSync(this->Softmax(logits_on_device,
  //       this->temperature_arr_));
  //     }
  //   }

  //   // perform sampling
  //   auto tstart = std::chrono::high_resolution_clock::now();
  //   int next_token;
  //   if (gen_temperature < 1e-6f) {
  //     next_token = this->SampleFromLogitsOnCPU(gen_temperature, gen_top_p);
  //   } else {
  //     next_token = this->SampleFromProbOnCPU(gen_top_p);
  //   }
  //   auto tend = std::chrono::high_resolution_clock::now();
  //   this->sample_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
  //   return next_token;
  // }

  // /*!
  //  * \brief Sample output token from logits on device
  //  */
  // int32_t SampleTokenFromLogits(picojson::object generation_config = picojson::object()) {
  //   // prepare generation settings
  //   // the generation_config will not override the original config
  //   // since is only used for this generation
  //   double gen_temperature;
  //   double gen_repetition_penalty;
  //   double gen_presence_penalty;
  //   double gen_frequency_penalty;
  //   double gen_top_p;
  //   this->ReadGenerationConfig(generation_config, &gen_temperature, &this->temperature_arr_,
  //                              &gen_repetition_penalty, &gen_presence_penalty,
  //                              &gen_frequency_penalty, &gen_top_p);

  //   this->ApplySoftmaxWithTemperatureOnCPU(gen_temperature);

  //   // perform sampling
  //   auto tstart = std::chrono::high_resolution_clock::now();
  //   int next_token;
  //   if (gen_temperature < 1e-6f) {
  //     next_token = this->SampleFromLogitsOnCPU(gen_temperature, gen_top_p);
  //   } else {
  //     next_token = this->SampleFromProbOnCPU(gen_top_p);
  //   }
  //   auto tend = std::chrono::high_resolution_clock::now();
  //   this->sample_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
  //   return next_token;
  // }

  // Comparator for the priority queue that compares pairs by their first element (the value)
  struct CompareByFirst {
    constexpr bool operator()(std::pair<float, int> const& a,
                              std::pair<float, int> const& b) const noexcept {
      return a.first > b.first;
    }
  };

  // Function to find the indices of the top N values in an array
  std::vector<int> findTopIndices(float* data, int size, int N) {
    // Using a min-heap to keep track of the top N values and their indices
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, CompareByFirst>
        pq;

    // Iterate over the array
    for (int i = 0; i < size; ++i) {
      // If we haven't found N elements yet, just add the current one
      if (pq.size() < N) {
        pq.push(std::make_pair(data[i], i));
      } else if (data[i] > pq.top().first) {
        // If the current element is greater than the smallest element in the heap, replace it
        pq.pop();
        pq.push(std::make_pair(data[i], i));
      }
    }

    // Extract the indices from the priority queue
    std::vector<int> indices;
    while (!pq.empty()) {
      // The indices are stored as the second element of the pair
      indices.push_back(pq.top().second);
      pq.pop();
    }

    // The indices will be in reverse order, so reverse them to get the top N in descending order
    std::reverse(indices.begin(), indices.end());
    return indices;
  }

  /*!
   * \brief Sample output token from logits on device
   */
  int32_t SampleTokenFromLogits(NDArray logits_on_device,
                                picojson::object generation_config = picojson::object()) {
    // prepare generation settings
    // the generation_config will not override the original config
    // since is only used for this generation
    double gen_temperature;
    double gen_repetition_penalty;
    double gen_presence_penalty;
    double gen_frequency_penalty;
    double gen_top_p;
    this->ReadGenerationConfig(generation_config, &gen_temperature, &this->temperature_arr_,
                               &gen_repetition_penalty, &gen_presence_penalty,
                               &gen_frequency_penalty, &gen_top_p);

    // print the shape of logits_on_device
    // std::cout << logits_on_device->shape[0] << ", " << logits_on_device->shape[1] << ", " << logits_on_device->shape[2] << std::endl;
    // std::cout << "temperature: " << gen_temperature << std::endl;

    // ***** the gpu code  this->UpdateLogitsOrProbOnCPUSync(this->Softmax(logits_on_device, this->temperature_arr_));
    // ***** is not working.  Whereas the cpu code this->ApplySoftmaxWithTemperatureOnCPU(gen_temperature); is working.
    // if (gen_temperature < 1e-6f) {
    //   this->UpdateLogitsOrProbOnCPUSync(logits_on_device);
    // } else {
    //   this->UpdateLogitsOrProbOnCPUSync(this->Softmax(logits_on_device, this->temperature_arr_));
    // }
    this->ApplySoftmaxWithTemperatureOnCPU(gen_temperature);

    //******DEBUG********

    // print the top 10 values of the logits
    //float* data = static_cast<float*>(logits_on_cpu_->data);

    // Find the indices of the top 10 values
    //std::vector<int> topIndices = findTopIndices(data, logits_on_cpu_.Shape()[2], 10);

    // Print the indices and the corresponding values
    // std::cout << "Top 10 values and their indices:" << std::endl;
    // for (int index : topIndices) {
    //   std::cout << "Value: " << data[index] << " at index: " << index << std::endl;
    // }

    //******DEBUG********

    // perform sampling
    auto tstart = std::chrono::high_resolution_clock::now();
    int next_token;
    if (gen_temperature < 1e-6f) {
      next_token = this->SampleFromLogitsOnCPU(gen_temperature, gen_top_p);
    } else {
      next_token = this->SampleFromProbOnCPU(gen_top_p);
    }
    auto tend = std::chrono::high_resolution_clock::now();
    this->sample_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    return next_token;
  }

  NDArray Softmax(NDArray input, NDArray temperature_arr) {
    NDArray ret;
    float* data = static_cast<float*>(temperature_arr->data);
    // iterating over the data

    std::cout << data[0] << __func__ << ':' << __LINE__ << std::endl;
    ret = ft_.softmax_func_(input, temperature_arr);
    return ret;
  }

  void ApplySoftmaxWithTemperatureOnCPU(float temperature) {
    CHECK(logits_on_cpu_.defined()) << "Logits on CPU not defined!";
    CHECK(logits_on_cpu_.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
    int vocab_size = logits_on_cpu_->shape[logits_on_cpu_->ndim - 1];
    float* logits_raw_data = static_cast<float*>(logits_on_cpu_->data);
    float m = std::numeric_limits<float>::min();
    float inv_temp = 1.0f / temperature;
    double d = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
      float x = logits_raw_data[i] * inv_temp;
      float m_prev = m;
      m = std::max(m, x);
      d = d * std::exp(m_prev - m) + std::exp(x - m);
    }
    // After computing 'd', check if it is zero
    if (d == 0.0) {
      // Handle the case where all logits are -inf
      // For example, you can set all softmax outputs to 1/vocab_size
      float uniform_prob = 1.0f / vocab_size;
      for (int i = 0; i < vocab_size; ++i) {
        logits_raw_data[i] = uniform_prob;
      }
    } else {
      // Normalize the logits as before
      for (int i = 0; i < vocab_size; ++i) {
        float x = logits_raw_data[i] * inv_temp;
        logits_raw_data[i] = std::exp(x - m) / d;
      }
    }
  }

  void UpdateLogitsOrProbOnCPUSync(NDArray logits_or_prob) {
    if (!logits_on_cpu_.defined()) {
      logits_on_cpu_ = logits_or_prob.CopyTo(DLDevice{kDLCPU, 0});
    } else {
      ICHECK_EQ(logits_on_cpu_->shape[0], logits_or_prob->shape[0])
          << "Expect size of logits remain unchanged";
      ICHECK_EQ(logits_on_cpu_->shape[1], logits_or_prob->shape[1])
          << "Expect size of logits remain unchanged";
      ICHECK_EQ(logits_on_cpu_->shape[2], logits_or_prob->shape[2])
          << "Expect size of logits remain unchanged";
      // clear the logits_on_cpu_ before copying
      logits_on_cpu_.CopyFrom(logits_or_prob);
    }
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
  }

  // Clear kv cache
  void ResetKVCache() { ft_.reset_kv_cache_func_(kv_cache_); }

  // Utils
  static double GetRandomNumber() { return RandomGenerator::GetInstance().GetRandomNumber(); }

  int32_t SampleFromLogitsOnCPU(float temperature, float top_p) {
    ICHECK(logits_on_cpu_.defined()) << "logits_on_cpu_ is not defined";
    ICHECK_EQ(logits_on_cpu_->ndim, 3) << "logits_on_cpu_ should be 3D";
    ICHECK_EQ(logits_on_cpu_->shape[0], 1) << "logits_on_cpu_ should be 1 batch";
    return fsample_topp_from_logits_(logits_on_cpu_, temperature, top_p, GetRandomNumber());
  }

  int32_t SampleFromProbOnCPU(float top_p) {
    ICHECK(logits_on_cpu_.defined()) << "logits_on_cpu_ is not defined";
    ICHECK_EQ(logits_on_cpu_->ndim, 3) << "logits_on_cpu_ should be 3D";
    ICHECK_EQ(logits_on_cpu_->shape[0], 1) << "logits_on_cpu_ should be 1 batch";
    return fsample_topp_from_prob_(logits_on_cpu_, top_p, GetRandomNumber());
  }

  //----------------------------
  // Statistics
  //----------------------------
  bool reset_stats_per_prefill_ = true;
  double encode_total_time = 0;
  double decode_total_time = 0;
  double sample_total_time = 0;
  double prefill_total_time = 0;
  int64_t decode_total_tokens = 0;
  int64_t prefill_total_tokens = 0;
  //---------------------------
  // Translation
  //---------------------------
  // total sequence len,
  int64_t total_seq_len_{0};
  // max window size, mean and max generation length, sliding window
  // If we use sliding window, max window size is its default max() value
  int64_t max_window_size_{std::numeric_limits<int64_t>::max()}, mean_gen_len_{128},
      max_gen_len_{512}, sliding_window_size_{-1}, prefill_chunk_size_{-1}, attention_sink_size_{0};
  // size of the vocab table
  int64_t vocab_size_;
  // number of shards in distributed inference
  int64_t num_shards_;
  // Load weights that were saved in sharded form
  bool use_presharded_weights_;
  // temperature
  double temperature_{0.8};
  // pre-allocated ndarray for temperature
  NDArray temperature_arr_;
  // top_p
  double top_p_{0.95};
  // output ids till now (refresh after encoding step)
  std::vector<int32_t> output_ids_;
  // stop tokens
  std::vector<int32_t> stop_tokens_;
  // Whether encounter stop str
  bool stop_triggered_{false};
  // output message till now (refresh after encoding step)
  std::string output_message_;
  // max target length
  int64_t max_target_len_{128};
  // max source length
  int64_t max_source_len_{128};
  // input sequence length
  int64_t input_seq_len_{0};
  // suppress tokens
  std::vector<int64_t> suppress_tokens_;
  // begin suppress tokens
  std::vector<int64_t> begin_suppress_tokens_;
  // forced tokens
  std::vector<std::pair<int, std::optional<int>>> forced_tokens_;

  // all_encoder_key_value
  ObjectRef all_encoder_key_value_{nullptr};

  // forced decoder map
  std::unordered_map<int64_t, int64_t> forced_decoder_map_;

  // embedding
  NDArray embedding_{nullptr};

  //----------------------------
  // Tokenizer
  //----------------------------
  // internal tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;
  // bos token
  int32_t bos_token_id_{1};
  // eos token id
  int32_t eos_token_id_{2};
  //----------------------------
  // TVM related states
  //----------------------------
  // runtime device
  Device device_;

  FunctionTable ft_;
  // sample top p from logits
  PackedFunc fsample_topp_from_logits_;
  // sample top p from prob
  PackedFunc fsample_topp_from_prob_;
  // input token id for encoder
  NDArray input_token_ids_{nullptr};
  // input token id for decoder
  NDArray output_token_ids_{nullptr};
  // local params
  ObjectRef params_;
  // KV cache
  ObjectRef kv_cache_;
  // Temp logits on cpu
  NDArray logits_on_cpu_{nullptr};
  // pre-allocated ndarray for decode function's input tokens
  DRef input_tokens_decode_{nullptr};
};

/*!
 * \brief A translate module implementation that exposes
 *  the functions as tvm::runtime::Module.
 *
 * We do it so that the module is accessible to any
 * language that tvm runtime can access.
 */

class TranslateModule : public ModuleNode {
 public:
  // clear global memory manager
  static void ClearGlobalMemoryManager() {
    // Step 0. Clear the previously allocated memory.
    const PackedFunc* fclear_memory_manager =
        tvm::runtime::Registry::Get("vm.builtin.memory_manager.clear");
    ICHECK(fclear_memory_manager) << "Cannot find env function vm.builtin.memory_manager.clear";
    (*fclear_memory_manager)();
  }

  // overrides
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "reload") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        translate_ = nullptr;
        ClearGlobalMemoryManager();
        translate_ = std::make_unique<TransformerTranslate>(TransformerTranslate(device_));
        ICHECK(2 <= args.size() && args.size() <= 3);
        if (args.size() == 2) {
          // args: reload_lib, model_path
          translate_->Reload(args[0], args[1]);
        } else if (args.size() == 3) {
          // args: reload_lib, model_path, app_config_json (used for overriding config)
          translate_->Reload(args[0], args[1], args[2]);
        }
      });
    } else if (name == "unload") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        translate_ = nullptr;
        ClearGlobalMemoryManager();
      });
    } else if (name == "encode") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK(1 <= args.size() && args.size() <= 4);
        if (args.size() == 3) {
          int type_code = args[1];
          if (type_code == kTVMStr) {
            // args: NDArray, typecode, app_config_json
            GetTranslate()->EncodeStep(static_cast<String>(args[0]), args[1], args[2]);
          } else {
            // args: NDArray, typecode, app_config_json
            GetTranslate()->EncodeStep(static_cast<tvm::runtime::NDArray>(args[0]), args[1],
                                       args[2]);
          }
        }
      });
    } else if (name == "decode") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK(0 <= args.size() && args.size() <= 1);
        if (args.size() == 0) {
          // args: generation_config_str = ""
          GetTranslate()->DecodeStep();
        } else if (args.size() == 1) {
          // args: generation_config_str
          GetTranslate()->DecodeStep(static_cast<String>(args[0]));
        }
      });
    } else if (name == "prefill") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK(0 <= args.size() && args.size() <= 1);
        if (args.size() == 0) {
          // args: generation_config_str = ""
          GetTranslate()->PrefillStep();
        } else if (args.size() == 1) {
          // args: generation_config_str
          GetTranslate()->PrefillStep(static_cast<String>(args[0]));
        }
      });
    } else if (name == "generate") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK(args.size() == 3);
        // args: NDArray, type_code, app_config_json
        GetTranslate()->Generate(args[0], args[1], args[2]);
        *rv = GetTranslate()->GetOutputMessage();
      });
    } else if (name == "get_message") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK(args.size() == 0);
        *rv = GetTranslate()->GetOutputMessage();
      });
    } else if (name == "stopped") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetTranslate()->CheckStopCondition();
      });
    } else if (name == "load_json_override") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 2);
        std::string config_str = args[0];
        bool partial_update = args[1];
        GetTranslate()->LoadJSONOverride(config_str, partial_update);
      });
    } else if (name == "runtime_stats_text") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetTranslate()->RuntimeStatsText();
      });
    } else if (name == "verbose_runtime_stats_text") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetTranslate()->VerboseRuntimeStatsText();
      });
    } else if (name == "reset_runtime_stats") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        GetTranslate()->ResetRuntimeStats();
      });
    } else if (name == "get_config_json") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetTranslate()->GetConfigJSON();
      });
    } else {
      return PackedFunc(nullptr);
    }
  }
  void Init(DLDevice device) { device_ = device; }

  TransformerTranslate* GetTranslate() {
    ICHECK(translate_ != nullptr) << "Chat is not initialized via reload";
    return translate_.get();
  }
  const char* type_key() const final { return "mlc.llm_translate"; }

 private:
  std::unique_ptr<TransformerTranslate> translate_ = nullptr;
  DLDevice device_;
};

tvm::runtime::Module CreateTranslateModule(DLDevice device) {
  ObjectPtr<TranslateModule> n = make_object<TranslateModule>();
  n->Init(device);
  return Module(n);
}

// register as a system function that can be queried
TVM_REGISTER_GLOBAL("mlc.llm_translate_create").set_body_typed([](int device_type, int device_id) {
  return CreateTranslateModule(DLDevice{static_cast<DLDeviceType>(device_type), device_id});
});

}  // namespace llm
}  // namespace mlc
