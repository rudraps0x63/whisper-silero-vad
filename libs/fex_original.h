#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <thread>
#include <algorithm>
#include <fstream>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#define COMMON_SAMPLE_RATE 16000
#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT 400
#define WHISPER_HOP_LENGTH 160
#define WHISPER_CHUNK_SIZE 30
#define WHISPER_N_MELS 80
#define SIN_COS_N_COUNT WHISPER_N_FFT

static float sin_vals[SIN_COS_N_COUNT];
static float cos_vals[SIN_COS_N_COUNT];

struct whisper_filters
{
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

struct whisper_mel
{
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

bool isNotZero(const std::vector<float> &vec)
{
    return std::all_of(vec.begin(), vec.end(), [](int i)
                       { return i != 0; });
}

static void fill_sin_cos_table()
{
    static bool is_filled = false;
    if (is_filled)
        return;
    for (int i = 0; i < SIN_COS_N_COUNT; i++)
    {
        double theta = (2 * M_PI * i) / SIN_COS_N_COUNT;
        sin_vals[i] = sinf(theta);
        cos_vals[i] = cosf(theta);
    }
    is_filled = true;
}

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
static void dft(const std::vector<float> &in, std::vector<float> &out)
{
    int N = in.size();

    out.resize(N * 2);
    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < N; k++)
    {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++)
        {
            int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT); // t = 2*M_PI*k*n/N
            re += in[n] * cos_vals[idx];                          // cos(t)
            im -= in[n] * sin_vals[idx];                          // sin(t)
        }

        out[k * 2 + 0] = re;
        out[k * 2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
static void fft(const std::vector<float> &in, std::vector<float> &out)
{
    out.resize(in.size() * 2);

    int N = in.size();

    if (N == 1)
    {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    if (N % 2 == 1)
    {
        dft(in, out);
        return;
    }

    std::vector<float> even;
    std::vector<float> odd;

    even.reserve(N / 2);
    odd.reserve(N / 2);

    for (int i = 0; i < N; i++)
    {
        if (i % 2 == 0)
        {
            even.push_back(in[i]);
        }
        else
        {
            odd.push_back(in[i]);
        }
    }

    std::vector<float> even_fft;
    std::vector<float> odd_fft;

    fft(even, even_fft);
    fft(odd, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;
    for (int k = 0; k < N / 2; k++)
    {
        int idx = k * sin_cos_step; // t = 2*M_PI*k/N
        float re = cos_vals[idx];   // cos(t)
        float im = -sin_vals[idx];  // sin(t)

        float re_odd = odd_fft[2 * k + 0];
        float im_odd = odd_fft[2 * k + 1];

        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + N / 2) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + N / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

static bool hann_window(int length, bool periodic, std::vector<float> &output)
{
    if (output.size() < static_cast<size_t>(length))
    {
        output.resize(length);
    }
    int offset = -1;
    if (periodic)
    {
        offset = 0;
    }
    for (int i = 0; i < length; i++)
    {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }

    return true;
}

static void log_mel_spectrogram_worker_thread(int ith, const std::vector<float> &hann, const std::vector<float> &samples,
                                              int n_samples, int frame_size, int frame_step, int n_threads,
                                              const whisper_filters &filters, whisper_mel &mel)
{
    std::vector<float> fft_in(frame_size, 0.0);
    std::vector<float> fft_out(2 * frame_step);
    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    int n_fft = 1 + (frame_size / 2);
    int i = ith;

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads)
    {
        const int offset = i * frame_step;

        // apply Hanning window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++)
        {
            fft_in[j] = hann[j] * samples[offset + j];
        }
        // fill the rest with zeros
        if (n_samples - offset < frame_size)
        {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(fft_in, fft_out);

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < frame_size; j++)
        {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++)
        {
            double sum = 0.0;

            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4)
            {
                sum +=
                    fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                    fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                    fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                    fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }

            // handle n_fft remainder
            for (; k < n_fft; k++)
            {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }

            sum = log10(std::max(sum, 1e-10));

            mel.data[j * mel.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads)
    {
        for (int j = 0; j < mel.n_mel; j++)
        {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
}

// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
static bool log_mel_spectrogram(
    const float *samples,
    const int n_samples,
    const int /*sample_rate*/,
    const int frame_size,
    const int frame_step,
    const int n_mel,
    const int n_threads,
    const whisper_filters &filters,
    const bool debug,
    whisper_mel &mel)
{
    // const int64_t t_start_us = ggml_time_us();

    // Hanning window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    std::vector<float> hann;
    hann_window(frame_size, true, hann);
    // printf("hann window %d\n", hann.size());

    // Calculate the length of padding
    int64_t stage_1_pad = WHISPER_SAMPLE_RATE * 30;
    int64_t stage_2_pad = frame_size / 2;

    // Initialize a vector and copy data from C array to it.
    std::vector<float> samples_padded;
    samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
    std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);

    // reflective pad 200 samples at the beginning of audio
    std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());

    mel.n_mel = n_mel;
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
    // Calculate number of frames + remove the last frame
    mel.n_len = (samples_padded.size() - frame_size) / frame_step;
    // Calculate semi-padded sample length to ensure compatibility
    mel.n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step;
    mel.data.resize(mel.n_mel * mel.n_len);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw)
        {
            workers[iw] = std::thread(
                log_mel_spectrogram_worker_thread, iw + 1, std::cref(hann), samples_padded,
                n_samples + stage_2_pad, frame_size, frame_step, n_threads,
                std::cref(filters), std::ref(mel));
        }

        // main thread
        log_mel_spectrogram_worker_thread(0, hann, samples_padded, n_samples + stage_2_pad, frame_size, frame_step, n_threads, filters, mel);

        for (int iw = 0; iw < n_threads - 1; ++iw)
        {
            workers[iw].join();
        }
    }

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel * mel.n_len; i++)
    {
        if (mel.data[i] > mmax)
        {
            mmax = mel.data[i];
        }
    }

    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel * mel.n_len; i++)
    {
        if (mel.data[i] < mmax)
        {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0) / 4.0;
    }

    // wstate.t_mel_us += ggml_time_us() - t_start_us;

    // Dump log_mel_spectrogram
    if (debug)
    {
        std::ofstream outFile("log_mel_spectrogram.json");
        outFile << "[";
        for (uint64_t i = 0; i < mel.data.size() - 1; i++)
        {
            outFile << mel.data[i] << ", ";
        }
        outFile << mel.data[mel.data.size() - 1] << "]";
        outFile.close();
    }

    return true;
}

bool read_wav(const std::string &fname, std::vector<float> &pcmf32, std::vector<std::vector<float>> &pcmf32s, bool stereo)
{
    drwav wav;
    std::vector<uint8_t> wav_data; // used for pipe input from stdin

    if (fname == "-")
    {
        {
            uint8_t buf[1024];
            while (true)
            {
                const size_t n = fread(buf, 1, sizeof(buf), stdin);
                if (n == 0)
                {
                    break;
                }
                wav_data.insert(wav_data.end(), buf, buf + n);
            }
        }

        if (drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr) == false)
        {
            fprintf(stderr, "error: failed to open WAV file from stdin\n");
            return false;
        }

        fprintf(stderr, "%s: read %zu bytes from stdin\n", __func__, wav_data.size());
    }
    else if (fname.size() > 256 || fname.size() > 40 && fname.substr(0, 4) == "RIFF" && fname.substr(8, 4) == "WAVE")
    {
        if (drwav_init_memory(&wav, fname.c_str(), fname.size(), nullptr) == false)
        {
            fprintf(stderr, "error: failed to open WAV file from fname buffer\n");
            return false;
        }
    }
    else if (drwav_init_file(&wav, fname.c_str(), nullptr) == false)
    {
        fprintf(stderr, "error: failed to open '%s' as WAV file\n", fname.c_str());
        return false;
    }

    if (wav.channels != 1 && wav.channels != 2)
    {
        fprintf(stderr, "%s: WAV file '%s' must be mono or stereo\n", __func__, fname.c_str());
        return false;
    }

    if (stereo && wav.channels != 2)
    {
        fprintf(stderr, "%s: WAV file '%s' must be stereo for diarization\n", __func__, fname.c_str());
        return false;
    }

    if (wav.sampleRate != COMMON_SAMPLE_RATE)
    {
        fprintf(stderr, "%s: WAV file '%s' must be %i kHz\n", __func__, fname.c_str(), COMMON_SAMPLE_RATE / 1000);
        return false;
    }

    if (wav.bitsPerSample != 16)
    {
        fprintf(stderr, "%s: WAV file '%s' must be 16-bit\n", __func__, fname.c_str());
        return false;
    }

    const uint64_t n = wav_data.empty() ? wav.totalPCMFrameCount : wav_data.size() / (wav.channels * wav.bitsPerSample / 8);

    std::vector<int16_t> pcm16;
    pcm16.resize(n * wav.channels);
    drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
    drwav_uninit(&wav);

    // convert to mono, float
    pcmf32.resize(n);
    if (wav.channels == 1)
    {
        for (uint64_t i = 0; i < n; i++)
        {
            pcmf32[i] = float(pcm16[i]) / 32768.0f;
        }
    }
    else
    {
        for (uint64_t i = 0; i < n; i++)
        {
            pcmf32[i] = float(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0f;
        }
    }

    if (stereo)
    {
        // convert to stereo, float
        pcmf32s.resize(2);

        pcmf32s[0].resize(n);
        pcmf32s[1].resize(n);
        for (uint64_t i = 0; i < n; i++)
        {
            pcmf32s[0][i] = float(pcm16[2 * i]) / 32768.0f;
            pcmf32s[1][i] = float(pcm16[2 * i + 1]) / 32768.0f;
        }
    }

    return true;
}


bool readWAVFromBuffer(const std::vector<uint8_t> &wav_data,
                       const std::string& fname,
                       std::vector<float> &pcmf32,
                       std::vector<std::vector<float>> &pcmf32s,
                       bool stereo)
{
    drwav wav;
    // std::vector<uint8_t> wav_data; // used for pipe input from stdin

    {
        // {
        //     uint8_t buf[1024];
        //     while (true)
        //     {
        //         const size_t n = fread(buf, 1, sizeof(buf), stdin);
        //         if (n == 0)
        //         {
        //             break;
        //         }
        //         wav_data.insert(wav_data.end(), buf, buf + n);
        //     }
        // }

        if (drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr) == false)
        {
            fprintf(stderr, "error: failed to open WAV file of %d bytes from stdin\n", (int)wav_data.size());
            return false;
        }

        //fprintf(stderr, "%s: read %zu bytes from stdin\n", __func__, wav_data.size());
    }
    // else if (fname.size() > 256 || fname.size() > 40 && fname.substr(0, 4) == "RIFF" && fname.substr(8, 4) == "WAVE")
    // {
    //     if (drwav_init_memory(&wav, fname.c_str(), fname.size(), nullptr) == false)
    //     {
    //         fprintf(stderr, "error: failed to open WAV file from fname buffer\n");
    //         return false;
    //     }
    // }
    // else if (drwav_init_file(&wav, fname.c_str(), nullptr) == false)
    // {
    //     fprintf(stderr, "error: failed to open '%s' as WAV file\n", fname.c_str());
    //     return false;
    // }

    if (wav.channels != 1 && wav.channels != 2)
    {
        fprintf(stderr, "%s: WAV file '%s' must be mono or stereo\n", __func__, fname.c_str());
        return false;
    }

    if (stereo && wav.channels != 2)
    {
        fprintf(stderr, "%s: WAV file '%s' must be stereo for diarization\n", __func__, fname.c_str());
        return false;
    }

    if (wav.sampleRate != COMMON_SAMPLE_RATE)
    {
        fprintf(stderr, "%s: WAV file '%s' must be %i kHz\n", __func__, fname.c_str(), COMMON_SAMPLE_RATE / 1000);
        return false;
    }

    if (wav.bitsPerSample != 16)
    {
        fprintf(stderr, "%s: WAV file '%s' must be 16-bit\n", __func__, fname.c_str());
        return false;
    }

    const uint64_t n = wav_data.empty() ? wav.totalPCMFrameCount : wav_data.size() / (wav.channels * wav.bitsPerSample / 8);

    std::vector<int16_t> pcm16;
    pcm16.resize(n * wav.channels);
    drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
    drwav_uninit(&wav);

    // convert to mono, float
    pcmf32.resize(n);
    if (wav.channels == 1)
    {
        for (uint64_t i = 0; i < n; i++)
        {
            pcmf32[i] = float(pcm16[i]) / 32768.0f;
        }
    }
    else
    {
        for (uint64_t i = 0; i < n; i++)
        {
            pcmf32[i] = float(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0f;
        }
    }

    if (stereo)
    {
        // convert to stereo, float
        pcmf32s.resize(2);

        pcmf32s[0].resize(n);
        pcmf32s[1].resize(n);
        for (uint64_t i = 0; i < n; i++)
        {
            pcmf32s[0][i] = float(pcm16[2 * i]) / 32768.0f;
            pcmf32s[1][i] = float(pcm16[2 * i + 1]) / 32768.0f;
        }
    }

    return true;
}


// read pcm data
bool read_pcm_16b(const std::string &fname, std::vector<float> &pcmf32, std::vector<std::vector<float>> &pcmf32s, bool stereo)
{
    // Open the PCM file for reading in binary mode
    std::ifstream file(fname, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file" << std::endl;
        return false;
    }

    // Get the size of the file
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate the number of samples
    std::streamsize num_samples = size / sizeof(int16_t);
    std::streamsize num_samples_per_channel = stereo ? num_samples / 2 : num_samples;

    // Create a vector to hold the samples
    std::vector<int16_t> samples(num_samples);

    // Read the samples from the file
    if (!file.read(reinterpret_cast<char *>(samples.data()), size))
    {
        std::cerr << "Failed to read all samples" << std::endl;
        return false;
    }

    // Convert to float and normalize to range [-1.0, 1.0]
    pcmf32.reserve(num_samples_per_channel);
    if (stereo)
    {
        pcmf32s.resize(2);
        pcmf32s[0].reserve(num_samples_per_channel);
        pcmf32s[1].reserve(num_samples_per_channel);
    }

    for (std::streamsize i = 0; i < num_samples; ++i)
    {
        float normalized_sample = samples[i] / 32768.0f; // Normalize 16-bit range to [-1.0, 1.0]
        if (stereo)
        {
            pcmf32s[i % 2].push_back(normalized_sample);
        }
        else
        {
            pcmf32.push_back(normalized_sample);
        }
    }

    // If stereo, we need to de-interleave the channels
    if (stereo)
    {
        std::vector<float> left_channel(num_samples_per_channel);
        std::vector<float> right_channel(num_samples_per_channel);

        for (std::streamsize i = 0, j = 0; i < num_samples; i += 2, ++j)
        {
            left_channel[j] = pcmf32[i];
            right_channel[j] = pcmf32[i + 1];
        }

        pcmf32s[0] = std::move(left_channel);
        pcmf32s[1] = std::move(right_channel);
        pcmf32.clear(); // Clear the mono vector, as it's not needed for stereo
    }

    return true;
}

bool read_pcm_32b(const std::string &fname, std::vector<float> &pcmf32, std::vector<std::vector<float>> &pcmf32s, bool stereo)
{
    // Open the PCM file for reading in binary mode
    std::ifstream file(fname, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file" << std::endl;
        return false;
    }

    // Get the size of the file
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate the number of samples
    std::streamsize num_samples = size / sizeof(int32_t); // Use sizeof(int32_t) for 32-bit samples
    std::streamsize num_samples_per_channel = stereo ? num_samples / 2 : num_samples;

    // Create a vector to hold the samples
    std::vector<int32_t> samples(num_samples); // Use int32_t for 32-bit samples

    // Read the samples from the file
    if (!file.read(reinterpret_cast<char *>(samples.data()), size))
    {
        std::cerr << "Failed to read all samples" << std::endl;
        return false;
    }

    // Convert to float and normalize to range [-1.0, 1.0]
    pcmf32.reserve(num_samples_per_channel);
    if (stereo)
    {
        pcmf32s.resize(2);
        pcmf32s[0].reserve(num_samples_per_channel);
        pcmf32s[1].reserve(num_samples_per_channel);
    }

    for (std::streamsize i = 0; i < num_samples; ++i)
    {
        // Normalize 32-bit range to [-1.0, 1.0]
        float normalized_sample = samples[i] / static_cast<float>(INT32_MAX);
        if (stereo)
        {
            pcmf32s[i % 2].push_back(normalized_sample);
        }
        else
        {
            pcmf32.push_back(normalized_sample);
        }
    }

    // If stereo, de-interleave the channels
    if (stereo)
    {
        std::vector<float> left_channel(num_samples_per_channel);
        std::vector<float> right_channel(num_samples_per_channel);

        for (std::streamsize i = 0, j = 0; i < num_samples; i += 2, ++j)
        {
            left_channel[j] = pcmf32s[0][j];
            right_channel[j] = pcmf32s[1][j];
        }

        pcmf32s[0] = std::move(left_channel);
        pcmf32s[1] = std::move(right_channel);
        pcmf32.clear(); // Clear the mono vector, as it's not needed for stereo
    }

    return true;
}
bool read_mel_filters(const std::string fname, whisper_filters &mel_filters)
{
    // Open the binary file
    std::ifstream bin_file(fname, std::ios::binary);

    // Read the dimensions
    int n_mels, n_fft;
    bin_file.read(reinterpret_cast<char *>(&n_mels), sizeof(n_mels));
    bin_file.read(reinterpret_cast<char *>(&n_fft), sizeof(n_fft));

    mel_filters.n_mel = n_mels;
    mel_filters.n_fft = n_fft;

    // Allocate a vector to hold all the filter coefficients
    mel_filters.data.resize(n_mels * n_fft);

    // Read the Mel filter banks
    bin_file.read(reinterpret_cast<char *>(mel_filters.data.data()), mel_filters.data.size() * sizeof(float));

    // Check if we've reached the end of the file
    if (!bin_file)
    {
        std::cerr << "Error reading Mel filters from file." << std::endl;
        return false;
    }

    // Close the file
    bin_file.close();

    // Now mel_filters contains the Mel filter banks and can be accessed as needed
    // For example, to access the filter bank for the m-th Mel and the f-th frequency bin:
    // float value = mel_filters[m * n_fft + f];

    return true;
}

bool read_mel_data(const std::string fname, int64_t n_rows, int64_t n_frame)
{
    // const std::string filename = "/home/munusairam/softwares/programs/c++_projects/whisper-audio/data/mel_data_3D.bin";

    // Dimensions of the data
    int n_batch = 1;

    // Open the binary file for reading
    std::ifstream in(fname, std::ios::in | std::ios::binary);
    if (!in)
    {
        std::cerr << "Cannot open " << fname << " for reading\n";
        return false;
    }

    // Allocate 3D vector for storing the data
    std::vector<std::vector<std::vector<float>>> mel_data_3D(n_batch, std::vector<std::vector<float>>(n_rows, std::vector<float>(n_frame)));

    // Read data from the file
    for (auto &batch : mel_data_3D)
    {
        for (auto &row : batch)
        {
            in.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(float));
        }
    }

    in.close();

    // Now, mel_data_3D contains the data read from the file
    // You can process or use it as needed
    for (auto &batch : mel_data_3D)
    {
        for (auto &row : batch)
        {
            for (auto &value : row)
            {
                // printf("%f ", value);
            }
        }
    }
    return true;
}

std::vector<std::vector<std::vector<float>>>
featurex(const std::string& fname_input,
         std::vector<uint8_t>& wavData,
         bool readFromBuffer = false)
{
    const std::string whisper_filter_fname = "data/mel_filters.bin";
    // const std::string fname_input = "/home/munusairam/Downloads/test/audio_0.pcm";
    // const std::string fname_output = "/home/munusairam/softwares/programs/c++_projects/whisper-audio/data/audio_bin/mel_data_3D_0.bin";

    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    whisper_filters mel_filters;
    int frame_size = WHISPER_N_FFT;
    int frame_step = WHISPER_HOP_LENGTH;
    int n_mel = WHISPER_N_MELS;
    int n_threads = 1;
    float min_freq = 20.0f;
    float max_freq = 8000.0f;
    int sample_rate = WHISPER_SAMPLE_RATE;
    int n_frame = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SIZE / WHISPER_HOP_LENGTH;
    read_mel_filters(whisper_filter_fname, mel_filters);
    // printf("mel_filters n_mel %d\n", mel_filters.n_mel);
    // printf("mel_filters n_fft %d\n", mel_filters.n_fft);
    // printf("mel_filters %d\n", mel_filters.data.size());
    auto mel = whisper_mel();
    fill_sin_cos_table();

    bool stereo = false;
    bool result = false;

    if (readFromBuffer)
        result = readWAVFromBuffer(wavData, fname_input, pcmf32, pcmf32s, stereo);
    else
        result = read_wav(fname_input, pcmf32, pcmf32s, stereo);

    // bool result = read_pcm_32b(fname_input, pcmf32, pcmf32s, stereo);

    if (result)
    {
        log_mel_spectrogram(
            pcmf32.data(),
            pcmf32.size(),
            COMMON_SAMPLE_RATE,
            frame_size,
            frame_step,
            n_mel,
            n_threads,
            mel_filters,
            true,
            mel);
    }

    // printf("%d\n", result);
    // printf("%d\n", pcmf32.size());
    // printf("%d\n", pcmf32s.size());
    // printf("%d\n", mel.n_len);
    // printf("%d\n", mel.data.size());

    int n_rows = mel.n_mel;
    int n_cols = mel.n_len;
    std::vector<std::vector<float>> mel_data_2D(n_rows, std::vector<float>(n_frame));

    if (n_frame < n_cols)
    {
        int diff = n_cols - n_frame;
        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_cols; ++j)
            {
                if (j < n_cols - diff)
                {
                    mel_data_2D[i][j] = mel.data[i * n_cols + j];
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_frame; ++j)
            {
                mel_data_2D[i][j] = mel.data[i * n_cols + j];
            }
        }
    }

    int n_batch = 1;
    std::vector<std::vector<std::vector<float>>> mel_data_3D(n_batch, std::vector<std::vector<float>>(n_rows, std::vector<float>(n_frame)));

    if (!mel_data_3D.empty())
    {
        mel_data_3D.back() = mel_data_2D;
    }
    else
    {
        // printf("The 3D vector is empty.\n");
    }

    return mel_data_3D;

    // for (int i = 0; i < mel_data_3D.size(); ++i)
    // {
    //     for (int j = 0; j < mel_data_3D[i].size(); ++j)
    //     {
    //         for (int k = 0; k < mel_data_3D[i][j].size(); ++k)
    //         {
    //             printf("%f ", mel_data_3D[i][j][k]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // // write the mel_data_3D to a file
    // std::ofstream out(fname_output, std::ios::out | std::ios::binary);
    // if (!out)
    // {
    //     std::cerr << "Cannot open data.bin for writing\n";
    //     return 1;
    // }

    // for (const auto &mel_data_2D : mel_data_3D)
    // {
    //     for (const auto &row : mel_data_2D)
    //     {
    //         out.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
    //     }
    // }

    // out.close();
    // read_mel_data(fname_output, 80, 3000);
}
