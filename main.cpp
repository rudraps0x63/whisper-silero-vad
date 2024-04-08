#include "libs/callback-sink.h"
#include "libs/chunker.h"
#include "libs/logger.h"
#include "libs/fex.h"
#include "model/cpp/model.h"
#include "silero/silero-vad-iter.h"
// #include "libs/fex_original.h"

#include <queue>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <cassert>
#include <cstdint>
#include <chrono>
#include <fcntl.h>
#include <argparse/argparse.hpp>

using namespace Logger;

std::mutex g_mu;
std::queue<std::vector<uint8_t>> g_queue;

#define DEBUG() std::cerr << __func__ << ":" << __LINE__ << ": "

static void
onBufferListCbProducerThread(GstBufferList*, gpointer);


struct Context {
  GstElement* appsrc_;
  GstElement* sink_;
  MarsChunker* chunker_;

  Translator* translator_;
  FeatureExtractor* fex_;
  uint max_depth_ = 6;
  uint cur_depth_ = 0;
  std::vector<int16_t> previous_;

  Context(std::vector<std::string>& args)
  {
    sink_ = mars_callback_sink_new();
    mars_callback_sink_set_buffer_list_callback(MARS_CALLBACK_SINK(sink_),
                                                onBufferListCbProducerThread,
                                                this,
                                                nullptr);

    chunker_ = MARS_CHUNKER(g_object_new(MARS_TYPE_CHUNKER,
                                         "input", "mic",
                                         "sink", sink_,
                                         //"output", "data/output_%01d.wav",
                                         "rate", 16000,
                                         "muxer", "wavenc",
                                         "minimum-silence-time", 500000000,
                                         "maximum-chunk-time", 1000000000,
                                         nullptr));

    fex_ = new FeatureExtractor;

    translator_ = new Translator(args);
    Log() << "Model loaded! Listening for new buffers...\n\n";
  }
};


static std::string
format_token_string(const std::string& text)
{
  uint start = 0, end = 0;

  for (uint i = 0; i < text.size(); ++i) {
    if (text[i] == ' ' && !start)
      start = i;
    else if (text[i] == '<' && start)
      end = i;
  }

  /* If we have a sensible output, then start cannot equal end */
  if (start < end)
    return text.substr(start + 1, end - start - 1).c_str();

  return { };
};


static bool
collect_pcm16_samples(std::vector<uint8_t>& wav_data, std::vector<int16_t>& pcm16)
{
  drwav wav;
  uint64_t n;
  bool stereo = false;
  bool err = false;

  if (drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr) == false) {
    DEBUG() << "Failed to initialize memory for decoding\n";
    err = true;
  }

  if (wav.channels != 1 && wav.channels != 2) {
    DEBUG() << "WAV audio must be mono or stereo\n";
    err = true;
  }

  if (stereo && wav.channels != 2) {
    DEBUG() << "WAV audio must be stereo for diarization\n";
    err = true;
  }

  if (wav.sampleRate != 16000) {
    DEBUG() << "WAV audio must be 16000 hz\n";
    err = true;
  }

  if (wav.bitsPerSample != 16) {
    DEBUG() << "WAV audio must contain 16 bits/sample\n";
    err = true;
  }

  if (err)
    return err;

  n = wav_data.empty() ? wav.totalPCMFrameCount
                         : wav_data.size() / (wav.channels * wav.bitsPerSample / 8);

  pcm16.resize(n * wav.channels);
  drwav_read_pcm_frames_s16(&wav, n, pcm16.data());

  drwav_uninit(&wav);

  return true;
}


static void
onBufferListCbProducerThread(GstBufferList* buffers, gpointer user_data)
{
  Context* ctx = (Context*)user_data;
  uint len = 0;
  std::vector<uint8_t> data;
  static int idx = 0;
  bool writeToFile = false;
  std::fstream fileStream;

  if (writeToFile) {
    fileStream.open("data/output_" + std::to_string(idx++) + ".wav", std::ios::binary | std::ios::trunc | std::ios::out);

    if (!fileStream.is_open())
      DEBUG() << "Couldn't open file for writing...\n";
  }

  assert(GST_IS_BUFFER_LIST(buffers));

  len = gst_buffer_list_length(buffers);

  // DEBUG() << "Got some buffers... " << len << "\n";

  for (int i = 0; i < len; ++i) {
    GstBuffer* buffer = gst_buffer_list_get(buffers, i);
    GstMapInfo mapInfo;

    if (!gst_buffer_map(buffer, &mapInfo, GST_MAP_READ)) {
      DEBUG() << "Couldn't map buffer #" << i + 1 << "! Continuing...\n";
      continue;
    }

    for (uint byteOffset = 0; byteOffset < mapInfo.size; ++byteOffset) {
      uint8_t bufferByte = *(mapInfo.data + byteOffset);

      data.push_back(bufferByte);

      if (writeToFile)
        fileStream << bufferByte;
    }

    gst_buffer_unmap(buffer, &mapInfo);
  }

  g_mu.lock();
  g_queue.push(data);
  g_mu.unlock();
}


static void
processWAVFiles(Context* ctx)
{
  namespace fs = std::filesystem;

  std::string text;
  std::vector<uint8_t> wavData;
  std::vector<int16_t> pcm16;
  std::vector<float> newWavData;
  fs::path dirPath { "/home/rudra/testing/" };

  while (true) {
    if (g_queue.empty())
      continue;

    g_mu.lock();
    wavData = g_queue.front();
    g_queue.pop();
    g_mu.unlock();

    if (!collect_pcm16_samples(wavData, pcm16)) {
      DEBUG() << "Couldn't collect PCM data for this run, skipping...\n";
      return;
    }

    for (int i = 0; i < pcm16.size(); ++i)
      newWavData.push_back(static_cast<float>(pcm16[i]) / 32768);

    std::vector<int16_t> finalWavData = WriteToWAVVector("", newWavData);
    // std::vector<int16_t> finalWavData;

    // if ((ctx->cur_depth_ %= ctx->max_depth_) == 0) { // Fresh series, reset
    //   ctx->previous_ = { };
    // }

    // DEBUG() << "Current depth is: " << ctx->cur_depth_
    //         << " => collecting previous " << ctx->cur_depth_
    //         << " + 1 current PCM16 samples (total: " << ctx->cur_depth_ + 1 << ")\n";

    // for (int i = 0; i < pcm16.size(); ++i)
    //   ctx->previous_.push_back(pcm16[i]); // Append current run to previous runs of current series

    // ++ctx->cur_depth_;

    // //auto featureVec = featurex("", wavData, true);
    // auto featureVec = ctx->fex_->run(ctx->previous_);
    // text = ctx->translator_->generateFromFeatureVector(featureVec);

    // auto featureVec = ctx->fex_->run(pcm16);
    auto featureVec = ctx->fex_->run(finalWavData);
    text = ctx->translator_->generateFromFeatureVector(featureVec);

    Log() << "Got transcription as:\n" << format_token_string(text) << "\n\n";
  }


  return;


  for (const auto& dirEntry : fs::directory_iterator(dirPath)) {
    if (dirEntry.path().extension().string() == ".wav") {
      std::string text;
      bool readFromBuffer = true;
      std::string fname = dirEntry.path().string();
      std::vector<uint8_t> wavData;
      std::vector<std::vector<std::vector<float>>> featureVec;

      Log() << "Now processing file: " << fname << "...\n";

      if (!readFromBuffer) {
        // featureVec = featurex(fname, wavData, false);
        ;
      }
      else {
        std::ifstream fstream { fname, std::ios::binary };
        int i = 0;

        while (!fstream.eof()) {
          // std::cout << ++i << '\n';
          unsigned char byte;
          fstream >> byte;
          wavData.push_back((uint8_t)byte);
        }
        int fd;

        if ((fd = open(fname.c_str(), O_RDWR)) >= 0)
        {
            unsigned char c;

            while (read(fd, &c, 1) == 1)
              wavData.push_back((uint8_t)c);
        }
        else {
          Log(Level::FATAL) << "Failed to read file!";
          continue;
        }

        Log() << "Read " << wavData.size() << " bytes. Sending for feature ext...\n";
      }

      text = ctx->translator_->generateFromFeatureVector(featureVec);
      Log() << "Got transcription as:\n" << text << "\n\n";
    }
  }
}


static void
doInferenceFromFile(Context* ctx, std::string fname)
{
  std::vector<uint8_t> v;
  std::vector<int16_t> vv;

  Log() << "Got transcription for given WAV file as:\n"
        // << ctx->translator_->generateFromFeatureVector(featurex(fname, v)) << "\n\n";
        << ctx->translator_->generateFromFeatureVector(ctx->fex_->run(vv)) << "\n\n";
}


int main(int argc, char* argv[])
{
  Context* ctx;
  std::vector<std::string> args;


  assert(argc >= 4);

  gst_init(&argc, &argv);

  for (int i = 1; i < argc; ++i)
    args.push_back(argv[i]);

  ctx = new Context(args);
//  doInferenceFromFile(ctx, argv[4]);

//  return 0;


  if (true) {
    mars_chunker_play(ctx->chunker_);
    assert(mars_chunker_is_playing(ctx->chunker_));

    Log() << "Chunker running (via mic) in the bg...\n\n";

    processWAVFiles(ctx);

    return 0;
  }

  return 0;
}
