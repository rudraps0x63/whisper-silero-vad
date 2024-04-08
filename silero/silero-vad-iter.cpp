#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <limits>
#include <chrono>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cstdio>
#include <cstdarg>

#include <torch/script.h> // One-stop header.
#include <torch/types.h>

#if __cplusplus < 201703L
#include <memory>
#endif

#include "silero-vad-iter.h"

// #define __DEBUG_SPEECH_PROB___

class timestamp_t
{
public:
    int start;
    int end;

    // default + parameterized constructor
    timestamp_t(int start = -1, int end = -1)
        : start(start), end(end){};

    // assignment operator modifies object, therefore non-const
    timestamp_t &operator=(const timestamp_t &a)
    {
        start = a.start;
        end = a.end;
        return *this;
    };

    // equality comparison. doesn't modify object. therefore const.
    bool operator==(const timestamp_t &a) const
    {
        return (start == a.start && end == a.end);
    };
    std::string c_str()
    {
        // return std::format("timestamp {:08d}, {:08d}", start, end);
        return format("{start:%08d,end:%08d}", start, end);
    };

private:
    std::string format(const char *fmt, ...)
    {
        char buf[256];

        va_list args;
        va_start(args, fmt);
        const auto r = std::vsnprintf(buf, sizeof buf, fmt, args);
        va_end(args);

        if (r < 0)
            // conversion failed
            return {};

        const size_t len = r;
        if (len < sizeof buf)
            // we fit in the buffer
            return {buf, len};

#if __cplusplus >= 201703L
        // C++17: Create a string and write to its underlying array
        std::string s(len, '\0');
        va_start(args, fmt);
        std::vsnprintf(s.data(), len + 1, fmt, args);
        va_end(args);

        return s;
#else
        // C++11 or C++14: We need to allocate scratch memory
        auto vbuf = std::unique_ptr<char[]>(new char[len + 1]);
        va_start(args, fmt);
        std::vsnprintf(vbuf.get(), len + 1, fmt, args);
        va_end(args);

        return {vbuf.get(), len};
#endif
    };
};

class VadIterator
{
private:
    // OnnxRuntime resources
    torch::jit::script::Module module_; // LibTorch module
    torch::Device device;
    // std::shared_ptr<Ort::Session> session = nullptr;

private:
    // void init_engine_threads(int inter_threads, int intra_threads)
    // {
    //     // The method should be called in each thread/proc in multi-thread/proc work
    //     session_options.SetIntraOpNumThreads(intra_threads);
    //     session_options.SetInterOpNumThreads(inter_threads);
    //     session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // };

    // Initialize the LibTorch model
    void init_torch_model(const std::string &model_path)
    {
        try
        {
            module_ = torch::jit::load(model_path);
            device = torch::kCPU;
            module_.to(device);
        }
        catch (const c10::Error &e)
        {
            std::cerr << "Error loading the model: " << e.msg() << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void reset_states()
    {
        // //_h and _c are reset to zero
        // _h = torch::zeros(hc_node_dims, torch::kFloat32).to(device);
        // _c = torch::zeros(hc_node_dims, torch::kFloat32).to(device);

        triggered = false;
        temp_end = 0;
        current_sample = 0;

        prev_end = next_start = 0;

        speeches.clear();
        current_speech = timestamp_t();
    };

    void predict(const std::vector<float> &data)
    {

        // Convert the input data to a LibTorch tensor
        torch::Tensor input_tensor = torch::from_blob(const_cast<float *>(data.data()),
                                                      {1, static_cast<int64_t>(data.size())},
                                                      torch::kFloat32);

        // Prepare the inputs for the model
        std::vector<torch::jit::IValue> inputs = {input_tensor, sr};

        // Forward pass
        auto output = module_.forward(inputs).toTensor();

        // Assuming the first output is speech probability, and the next two are LSTM states which are
        // updated recursively
        float speech_prob = output.item<float>();
        //_h = output[1].toTensor();
        //_c = output[2].toTensor();
        // std::cout << speech_prob << " " << __func__ << " " << __LINE__ << std::endl;

        // Push forward sample index
        current_sample += window_size_samples;

        // Reset temp_end when > threshold
        if ((speech_prob >= threshold))
        {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
            printf("{    start: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            if (temp_end != 0)
            {
                temp_end = 0;
                if (next_start < prev_end)
                    next_start = current_sample - window_size_samples;
            }
            if (triggered == false)
            {
                triggered = true;

                current_speech.start = current_sample - window_size_samples;
            }
            return;
        }

        if (
            (triggered == true) && ((current_sample - current_speech.start) > max_speech_samples))
        {
            if (prev_end > 0)
            {
                current_speech.end = prev_end;
                speeches.push_back(current_speech);
                current_speech = timestamp_t();

                // previously reached silence(< neg_thres) and is still not speech(< thres)
                if (next_start < prev_end)
                    triggered = false;
                else
                {
                    current_speech.start = next_start;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
            }
            else
            {
                current_speech.end = current_sample;
                speeches.push_back(current_speech);
                current_speech = timestamp_t();
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
            }
            return;
        }
        if ((speech_prob >= (threshold - 0.15)) && (speech_prob < threshold))
        {
            if (triggered)
            {
#ifdef __DEBUG_SPEECH_PROB___
                float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
                printf("{ speeking: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            }
            else
            {
#ifdef __DEBUG_SPEECH_PROB___
                float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
                printf("{  silence: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            }
            return;
        }

        // 4) End
        if ((speech_prob < (threshold - 0.15)))
        {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples - speech_pad_samples; // minus window_size_samples to get precise start time point.
            printf("{      end: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            if (triggered == true)
            {
                if (temp_end == 0)
                {
                    temp_end = current_sample;
                }
                if (current_sample - temp_end > min_silence_samples_at_max_speech)
                    prev_end = temp_end;
                // a. silence < min_slience_samples, continue speaking
                if ((current_sample - temp_end) < min_silence_samples)
                {
                }
                // b. silence >= min_slience_samples, end speaking
                else
                {
                    current_speech.end = temp_end;
                    if (current_speech.end - current_speech.start > min_speech_samples)
                    {
                        speeches.push_back(current_speech);
                        current_speech = timestamp_t();
                        prev_end = 0;
                        next_start = 0;
                        temp_end = 0;
                        triggered = false;
                    }
                }
            }
            else
            {
                // may first windows see end state.
            }
            return;
        }
    };

public:
    void process(const std::vector<float> &input_wav)
    {
        reset_states();

        audio_length_samples = input_wav.size();

        for (int j = 0; j < audio_length_samples; j += window_size_samples)
        {
            if (j + window_size_samples > audio_length_samples)
                break;
            std::vector<float> r{&input_wav[0] + j, &input_wav[0] + j + window_size_samples};
            predict(r);
        }

        if (current_speech.start >= 0)
        {
            current_speech.end = audio_length_samples;
            speeches.push_back(current_speech);
            current_speech = timestamp_t();
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
        }
    };

    void process(const std::vector<float> &input_wav, std::vector<float> &output_wav)
    {
        process(input_wav);
        collect_chunks(input_wav, output_wav);
    }

    void collect_chunks(const std::vector<float> &input_wav, std::vector<float> &output_wav)
    {
        output_wav.clear();
        for (int i = 0; i < speeches.size(); i++)
        {
#ifdef __DEBUG_SPEECH_PROB___
            std::cout << speeches[i].c_str() << std::endl;
#endif // #ifdef __DEBUG_SPEECH_PROB___
            std::vector<float> slice(&input_wav[speeches[i].start], &input_wav[speeches[i].end]);
            output_wav.insert(output_wav.end(), slice.begin(), slice.end());
        }
    };

    const std::vector<timestamp_t> get_speech_timestamps() const
    {
        return speeches;
    }

    void drop_chunks(const std::vector<float> &input_wav, std::vector<float> &output_wav)
    {
        output_wav.clear();
        int current_start = 0;
        for (int i = 0; i < speeches.size(); i++)
        {

            std::vector<float> slice(&input_wav[current_start], &input_wav[speeches[i].start]);
            output_wav.insert(output_wav.end(), slice.begin(), slice.end());
            current_start = speeches[i].end;
        }

        std::vector<float> slice(&input_wav[current_start], &input_wav[input_wav.size()]);
        output_wav.insert(output_wav.end(), slice.begin(), slice.end());
    };

private:
    // model config
    int64_t window_size_samples; // Assign when init, support 256 512 768 for 8k; 512 1024 1536 for 16k.
    int sample_rate;             // Assign when init support 16000 or 8000
    int sr_per_ms;               // Assign when init, support 8 or 16
    float threshold;
    int min_silence_samples;               // sr_per_ms * #ms
    int min_silence_samples_at_max_speech; // sr_per_ms * #98
    int min_speech_samples;                // sr_per_ms * #ms
    float max_speech_samples;
    int speech_pad_samples; // usually a
    int audio_length_samples;

    // model states
    bool triggered = false;
    unsigned int temp_end = 0;
    unsigned int current_sample = 0;
    // MAX 4294967295 samples / 8sample per ms / 1000 / 60 = 8947 minutes
    int prev_end;
    int next_start = 0;

    // Output timestamp
    std::vector<timestamp_t> speeches;
    timestamp_t current_speech;

    // Onnx model
    // Inputs
    // std::vector<Ort::Value> ort_inputs;

    // std::vector<const char *> input_node_names = {"input", "sr", "h", "c"};
    torch::Tensor input;
    int64_t sr;
    // unsigned int size_hc = 2 * 1 * 64; // It's FIXED.
    // torch::Tensor _h;
    // torch::Tensor _c;

    int64_t input_node_dims[2] = {};
    const int64_t sr_node_dims[1] = {1};
    const int64_t hc_node_dims[3] = {2, 1, 64};

    // Outputs
    // std::vector<Ort::Value> ort_outputs;
    // std::vector<const char *> output_node_names = {"output", "hn", "cn"};

public:
    // Construction
    VadIterator(const std::string ModelPath,
                int Sample_rate = 16000, int windows_frame_size = 64,
                float Threshold = 0.5, int min_silence_duration_ms = 0,
                int speech_pad_ms = 64, int min_speech_duration_ms = 64,
                float max_speech_duration_s = std::numeric_limits<float>::infinity())
        : device(torch::kCPU) // // Initialize the device here
    {
        init_torch_model(ModelPath);
        threshold = Threshold;
        sample_rate = Sample_rate;
        sr_per_ms = sample_rate / 1000;

        window_size_samples = windows_frame_size * sr_per_ms;

        min_speech_samples = sr_per_ms * min_speech_duration_ms;
        speech_pad_samples = sr_per_ms * speech_pad_ms;

        max_speech_samples = (sample_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples);

        min_silence_samples = sr_per_ms * min_silence_duration_ms;
        min_silence_samples_at_max_speech = sr_per_ms * 98;

        input = torch::zeros(window_size_samples, torch::kFloat32);
        input_node_dims[0] = 1;
        input_node_dims[1] = window_size_samples;

        // // Declare the dimension of _h and _c
        // _h = torch::zeros(hc_node_dims, torch::kFloat32).to(device);
        // _c = torch::zeros(hc_node_dims, torch::kFloat32).to(device);
        sr = sample_rate;
    };
};

// debug purpose
float CalculateEnergy(const float *samples, int num_samples)
{
    float energy = 0.0f;
    for (int i = 0; i < num_samples; ++i)
    {
        energy += samples[i] * samples[i];
    }
    return energy;
}

// debug purpose
bool ContainsAudibleAudio(const float *samples, int num_samples, float silence_threshold)
{
    float energy = CalculateEnergy(samples, num_samples);
    return energy > silence_threshold;
}


std::vector<int16_t>
WriteToWAVVector(std::string& modelPath, std::vector<float>& data)
{
    std::vector<timestamp_t> stamps;

    // Read wav
    std::vector<float> input_wav(data.size());
    std::vector<float> output_wav;

    for (int i = 0; i < data.size(); i++)
    {
        input_wav[i] = static_cast<float>(*(data.data() + i));
    }

    // ========== Set the params ====================
    // Sample rate and other parameters should match your model and the audio files you're processing
    int sampleRate           = 16000; // Example sample rate
    int windowFrameSize      = 128;   // Adjust based on your model's requirements
    float threshold          = 0.5;   // Threshold for speech detection
    int minSilenceDurationMs = 0;     // Example minimum silence duration in milliseconds
    int speechPadMs          = 64;    // Speech padding
    int minSpeechDurationMs  = 200;   // Minimum speech duration in milliseconds
    float maxSpeechDurationS = 10.0;  // Maximum speech duration in seconds

    // Path to the LibTorch model (adjust the path to your actual model file)
    //std::string modelPath = "../data/silero-vad.pt";

    // ===== Test configs =====
    // Initialize the voice activity detection iterator with the model
    VadIterator vad(modelPath,
                    sampleRate,
                    windowFrameSize,
                    threshold,
                    minSilenceDurationMs,
                    speechPadMs,
                    minSpeechDurationMs,
                    maxSpeechDurationS);

    // ==============================================
    // ==== = Example 1 of full function  =====
    // ==============================================
    vad.process(input_wav);

    // 1.a get_speech_timestamps
    stamps = vad.get_speech_timestamps();
    for (int i = 0; i < stamps.size(); i++)
    {
        std::cout << stamps[i].c_str() << std::endl;
    }

    // 1.b collect_chunks output wav
    vad.collect_chunks(input_wav, output_wav);

    wav::WavWriter wavdata(output_wav.data(), output_wav.size(), 1, 16000, 16);

    return wavdata.Write();

    // 1.c drop_chunks output wav
    // vad.drop_chunks(input_wav, output_wav);
}

int WriteToWAVFile(int argc, char* argv[])
{
    // std::string filename;
    // std::vector<timestamp_t> stamps;

    // if (argc != 2)
    //     throw std::runtime_error("A filename to process is expected...");

    // filename = argv[1];

    // // Read wav
    // wav::WavReader wav_reader(filename); // 16000,1,32float
    // std::vector<float> input_wav(wav_reader.num_samples());
    // std::vector<float> output_wav;

    // for (int i = 0; i < wav_reader.num_samples(); i++)
    // {
    //     input_wav[i] = static_cast<float>(*(wav_reader.data() + i));
    // }

    // // ========== Set the params ====================
    // // Sample rate and other parameters should match your model and the audio files you're processing
    // int sampleRate = 16000;          // Example sample rate
    // int windowFrameSize = 128;        // Adjust based on your model's requirements
    // float threshold = 0.5;           // Threshold for speech detection
    // int minSilenceDurationMs = 0;    // Example minimum silence duration in milliseconds
    // int speechPadMs = 64;            // Speech padding
    // int minSpeechDurationMs = 200;   // Minimum speech duration in milliseconds
    // float maxSpeechDurationS = 10.0; // Maximum speech duration in seconds

    // // Path to the LibTorch model (adjust the path to your actual model file)
    // std::string modelPath = "../data/silero-vad.pt";

    // // ===== Test configs =====
    // // Initialize the voice activity detection iterator with the model
    // VadIterator vad(modelPath, sampleRate, windowFrameSize, threshold, minSilenceDurationMs, speechPadMs, minSpeechDurationMs, maxSpeechDurationS);

    // // ==============================================
    // // ==== = Example 1 of full function  =====
    // // ==============================================
    // vad.process(input_wav);

    // // 1.a get_speech_timestamps
    // stamps = vad.get_speech_timestamps();
    // for (int i = 0; i < stamps.size(); i++)
    // {

    //     std::cout << stamps[i].c_str() << std::endl;
    // }

    // // 1.b collect_chunks output wav
    // vad.collect_chunks(input_wav, output_wav);

    // // 1.c drop_chunks output wav
    // vad.drop_chunks(input_wav, output_wav);

    // // ==============================================
    // // ===== Example 2 of simple full function  =====
    // // ==============================================
    // vad.process(input_wav, output_wav);

    // stamps = vad.get_speech_timestamps();
    // for (int i = 0; i < stamps.size(); i++)
    // {

    //     std::cout << stamps[i].c_str() << std::endl;
    // }

    // // ==============================================
    // // ===== Example 3 of full function  =====
    // // ==============================================
    // for (int i = 0; i < 2; i++)
    //     vad.process(input_wav, output_wav);

    // // write the output wav to a wav file
    // wav::WavWriter wav_writer(output_wav.data(), output_wav.size(), 1, 16000, 16);
    // wav_writer.Write("../data/audio-samples/audio-vad.wav");

    return { };
}
