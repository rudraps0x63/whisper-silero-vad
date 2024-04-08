[discord-url]: https://discord.gg/9Xpy2HGBuD

# MLC LLM Whisper-tiny

## Install vulkan, gitlfs, tvm , mlc.

1. Install vulkan driver if your system has AMD RADEON Graphics(applicable for the laptop given by the company)

   ###### In arch linux:

   - sudo pacman -S mesa
   - sudo pacman -S vulkan-radeon
   - sudo pacman -S vulkan-tools vulkan-icd-loader

   ###### After installing these packages, reboot your system to ensure that the drivers are properly loaded. Once your system restarts, you can verify the Vulkan installation. Run: vulkaninfo

2. Install git-lfs for proper downloading of the model weights

   - arch linux: \
      sudo pacman -S git-lfs

3. Install mlc

   1. clone the repository \
      git clone https://github.com/mlc-ai/mlc-llm.git --recursive
   2. enter to root directory of the repo \
      cd mlc-llm
   3. create virtual env \
      python -m venv venv
   4. activate the venv and install the tvm in venv \
      source venv/bin/activate \
      pip install --pre mlc-ai-nightly -f https://mlc.ai/wheels
   5. if you want to compile the model and generate the .so file, install \
      pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu \
      pip install transformers datasets pysoundfile librosa ipykernel

4. Verify the mlc installation by \
   python -c "import mlc_chat; print(mlc_chat)"

## Build using c++

1. cd mlc-llm
2. mkdir -p build && cd build && python3 ../cmake/gen_cmake_config.py && cmake .. && cmake --build . --parallel $(nproc) && cd ..

## Create support for the whisper-tiny model

- Install whisper-tiny model

  - cd python
  - pip install .

- To check the mlc_chat created
  - mlc_chat --help

## Compile whisper model

1. Download the whisper-tiny from the huggingface in mlc-llm/dist/models/ \
   git clone https://huggingface.co/openai/whisper-tiny
2. create a directory mlc-llm/dist/libs/whipser-tiny to store the compiled model. \
   mkdir -p mlc-llm/dist/libs/whisper-tiny

3. Compile the model and generate the .so file

   1. Convert weight
      - cd mlc-llm/python \
      - mlc_chat convert_weight \
        --model-type whisper-tiny \
        ./dist/models/whisper-tiny \
        --quantization q0f32 \
        --output ./dist/libs/whisper-tiny
   2. Generate mlc_chat config
      - mlc_chat gen_config \
        --model-type whisper-tiny \
        ./dist/models/whisper-tiny/ \
        --quantization q0f32 \
        --conv-template whisper-tiny \
        --output ./dist/libs/whisper-tiny
   3. Compile the whisper-tiny model

      - mlc_chat compile \
        --model-type whisper-tiny \
        ./dist/libs/whisper-tiny/ \
        --quantization q0f32 \
        --device vulkan \
         --output ./dist/libs/whisper-tiny/whisper-tiny.so

   4. The above 3 commands are in the notebook mlc-llm/mlc_llm/test_mlc_chat_whisper.ipynb. Run the notebook in the kernel mlc-llm/venv.

## Execute sample example

./build/mlc_chat_cli --model dist/libs/whisper-tiny/ \
 --model-lib-path .dist/libs/whisper-tiny/whisper-tiny.so \
--translate --input-features-bin \
./data/audio_bin/mel_data_3D_1.bin

## Java script support using Bare

1.  clone Bare. \
    cd mlc-llm \
    git clone --recursive https://github.com/holepunchto/bare.git
2.  Build bare from source as given under section Building in Readme.md.
3.  Create config.cmake. \
    cd mlc-llm \
    python cmake/gen_cmake_config.py
4.  Build from source. \
    cd mlc-llm \
    rm -fr build \
    rm -fr prebuilds \
    bare-dev install
5.  Check for the .so files. \
    ldd -d prebuilds/linux-x64/mlc-llm.bare (for linux) \
    list of .so files: \
    - linux-vdso.so.1
    - libmlc_llm.so
    - libtvm_runtime.so
    - libstdc++.so.6
    - libm.so.6
    - libgcc_s.so.1
    - libc.so.6
    - libvulkan.so.1
    - libdl.so.2
    - libpthread.so.0
6.  bare index.js

# MLC LLM

[Documentation](https://llm.mlc.ai/docs) | [Blog](https://blog.mlc.ai/) | [Discord][discord-url]

**M**achine **L**earning **C**ompilation for **L**arge **L**anguage **M**odels (MLC LLM) is a high-performance universal deployment solution that allows native deployment of any large language models with native APIs with compiler acceleration. The mission of this project is to enable everyone to develop, optimize and deploy AI models natively on everyone's devices with ML compilation techniques.

**Universal deployment.** MLC LLM supports the following platforms and hardware:

<table style="width:100%">
  <thead>
    <tr>
      <th style="width:15%"> </th>
      <th style="width:20%">AMD GPU</th>
      <th style="width:20%">NVIDIA GPU</th>
      <th style="width:20%">Apple GPU</th>
      <th style="width:24%">Intel GPU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Linux / Win</td>
      <td>✅ Vulkan, ROCm</td>
      <td>✅ Vulkan, CUDA</td>
      <td>N/A</td>
      <td>✅ Vulkan</td>
    </tr>
    <tr>
      <td>macOS</td>
      <td>✅ Metal (dGPU)</td>
      <td>N/A</td>
      <td>✅ Metal</td>
      <td>✅ Metal (iGPU)</td>
    </tr>
    <tr>
      <td>Web Browser</td>
      <td colspan=4>✅ WebGPU and WASM </td>
    </tr>
    <tr>
      <td>iOS / iPadOS</td>
      <td colspan=4>✅ Metal on Apple A-series GPU</td>
    </tr>
    <tr>
      <td>Android</td>
      <td colspan=2>✅ OpenCL on Adreno GPU</td>
      <td colspan=2>✅ OpenCL on Mali GPU</td>
    </tr>
  </tbody>
</table>

**Scalable.** MLC LLM scales universally on NVIDIA and AMD GPUs, cloud and gaming GPUs. Below
showcases our single batch decoding performance with prefilling = 1 and decoding = 256.

Performance of 4-bit CodeLlama-34B and Llama2-70B on two NVIDIA RTX 4090 and two AMD Radeon 7900 XTX:

<p float="left">
  <img src="site/img/multi-gpu/figure-1.svg" width="40%"/>
  <img src="site/img/multi-gpu/figure-3.svg" width="30%"/>
</p>

Scaling of fp16 and 4-bit CodeLlama-34 and Llama2-70B on A100-80G-PCIe and A10G-24G-PCIe, up to 8 GPUs:

<p float="center">
  <img src="site/img/multi-gpu/figure-2.svg" width="100%"/>
</p>

## News

- [10/18/2023] [[Post]](https://blog.mlc.ai/2023/10/19/Scalable-Language-Model-Inference-on-Multiple-NVDIA-AMD-GPUs) Scalable multi-GPU support for CUDA and ROCm are official.
- [09/02/2023] Prebuilt ROCm 5.7 and CUDA 12.2 package is [available](https://llm.mlc.ai/docs/install/tvm.html#option-1-prebuilt-package).
- [08/25/2023] CodeLlama support is up.
- [08/14/2023] [[Post]](https://blog.mlc.ai/2023/08/09/GPU-Accelerated-LLM-on-Orange-Pi) Mali GPU support is up on Orange Pi.
- [08/09/2023] [[Post]](https://blog.mlc.ai/2023/08/09/Making-AMD-GPUs-competitive-for-LLM-inference) ROCm backend is mature to use.
- [08/02/2023] [Dockerfile](https://github.com/mlc-ai/llm-perf-bench/) is released for CUDA performance benchmarking.
- [07/19/2023] Support for Llama2-7B/13B/70B is up.
- [05/22/2023] [[Post]](https://blog.mlc.ai/2023/05/22/bringing-open-large-language-models-to-consumer-devices) RedPajama support is up.
- [05/08/2023] [[Post]](https://blog.mlc.ai/2023/05/08/bringing-hardware-accelerated-language-models-to-android-devices) MLC LLM is now available on Android.
- [05/01/2023] [[Post]](https://blog.mlc.ai/2023/05/01/bringing-accelerated-llm-to-consumer-hardware) MLC LLM is released with Metal, Vulkan and CUDA backends.
- [04/14/2023] [WebLLM](https://github.com/mlc-ai/web-llm) is released prior to MLC LLM with WebGPU and WebAssembly backend.

## Getting Started

Please visit our [documentation](https://llm.mlc.ai/docs/index.html#getting-started) for detailed instructions.

## Model Support

MLC LLM supports a wide range of model architectures and variants. We have the following prebuilts which you can
use off-the-shelf. Visit [Prebuilt Models](https://llm.mlc.ai/docs/prebuilt_models.html) to see the full list, and [Compile Models via MLC](https://llm.mlc.ai/docs/compilation/compile_models.html) to see how to use models not on this list.

<table style="width:100%">
  <thead>
    <tr>
      <th style="width:40%">Architecture</th>
      <th style="width:60%">Prebuilt Model Variants</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Llama</td>
      <td>Llama-2, Code Llama, Vicuna, WizardLM, WizardMath, OpenOrca Platypus2, FlagAlpha Llama-2 Chinese, georgesung Llama-2 Uncensored</td>
    </tr>
    <tr>
      <td>GPT-NeoX</td>
      <td>RedPajama</td>
    </tr>
    <tr>
      <td>GPT-J</td>
      <td></td>
    </tr>
    <tr>
      <td>RWKV</td>
      <td>RWKV-raven</td>
    </tr>
    <tr>
      <td>MiniGPT</td>
      <td></td>
    </tr>
    <tr>
      <td>GPTBigCode</td>
      <td>WizardCoder</td>
    </tr>
    <tr>
      <td>ChatGLM</td>
      <td></td>
    </tr>
    <tr>
      <td>StableLM</td>
      <td></td>
    </tr>
  </tbody>
</table>

## Universal Deployment APIs

MLC LLM provides multiple sets of APIs across platforms and environments. These include

- [Python API](https://llm.mlc.ai/docs/deploy/python.html)
- [OpenAI-compatible Rest-API](https://llm.mlc.ai/docs/deploy/rest.html)
- [C++ API](https://llm.mlc.ai/docs/deploy/cli.html)
- [JavaScript API](https://llm.mlc.ai/docs/deploy/javascript.html) and [Web LLM](https://github.com/mlc-ai/web-llm)
- [Swift API for iOS App](https://llm.mlc.ai/docs/deploy/ios.html)
- [Java API and Android App](https://llm.mlc.ai/docs/deploy/android.html)

## Citation

Please consider citing our project if you find it useful:

```bibtex
@software{mlc-llm,
    author = {MLC team},
    title = {{MLC-LLM}},
    url = {https://github.com/mlc-ai/mlc-llm},
    year = {2023}
}
```

The underlying techniques of MLC LLM include:

<details>
  <summary>References (Click to expand)</summary>
  
  ```bibtex
  @inproceedings{tensorir,
      author = {Feng, Siyuan and Hou, Bohan and Jin, Hongyi and Lin, Wuwei and Shao, Junru and Lai, Ruihang and Ye, Zihao and Zheng, Lianmin and Yu, Cody Hao and Yu, Yong and Chen, Tianqi},
      title = {TensorIR: An Abstraction for Automatic Tensorized Program Optimization},
      year = {2023},
      isbn = {9781450399166},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3575693.3576933},
      doi = {10.1145/3575693.3576933},
      booktitle = {Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2},
      pages = {804–817},
      numpages = {14},
      keywords = {Tensor Computation, Machine Learning Compiler, Deep Neural Network},
      location = {Vancouver, BC, Canada},
      series = {ASPLOS 2023}
  }

@inproceedings{metaschedule,
author = {Shao, Junru and Zhou, Xiyou and Feng, Siyuan and Hou, Bohan and Lai, Ruihang and Jin, Hongyi and Lin, Wuwei and Masuda, Masahiro and Yu, Cody Hao and Chen, Tianqi},
booktitle = {Advances in Neural Information Processing Systems},
editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
pages = {35783--35796},
publisher = {Curran Associates, Inc.},
title = {Tensor Program Optimization with Probabilistic Programs},
url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/e894eafae43e68b4c8dfdacf742bcbf3-Paper-Conference.pdf},
volume = {35},
year = {2022}
}

@inproceedings{tvm,
author = {Tianqi Chen and Thierry Moreau and Ziheng Jiang and Lianmin Zheng and Eddie Yan and Haichen Shen and Meghan Cowan and Leyuan Wang and Yuwei Hu and Luis Ceze and Carlos Guestrin and Arvind Krishnamurthy},
title = {{TVM}: An Automated {End-to-End} Optimizing Compiler for Deep Learning},
booktitle = {13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18)},
year = {2018},
isbn = {978-1-939133-08-3},
address = {Carlsbad, CA},
pages = {578--594},
url = {https://www.usenix.org/conference/osdi18/presentation/chen},
publisher = {USENIX Association},
month = oct,
}

```
</details>

## Links

- You might want to check out our online public [Machine Learning Compilation course](https://mlc.ai) for a systematic
walkthrough of our approaches.
- [WebLLM](https://webllm.mlc.ai/) is a companion project using MLC LLM's WebGPU and WebAssembly backend.
- [WebStableDiffusion](https://websd.mlc.ai/) is a companion project for diffusion models with the WebGPU backend.

```
