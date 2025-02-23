/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <gtest/gtest.h>

#include <flashinfer/decode.cuh>
#include <flashinfer/handler.cuh>
#include <type_traits>

#include "cpu_reference.h"
#include "utils.h"

using namespace flashinfer;

template <typename T>
void _TestBatchDecodingKernelCorrectness(size_t page_size, size_t batch_size, size_t num_qo_heads,
                                         size_t num_kv_heads, size_t head_dim,
                                         flashinfer::RotaryMode rotary_mode, bool cooperative) {
  std::vector<int32_t> seq_lens(batch_size);
  utils::vec_randint_(seq_lens, 1, 1024);
  std::vector<int32_t> append_indptr{0};
  for (size_t i = 0; i < batch_size; ++i) {
    append_indptr.push_back(append_indptr.back() + seq_lens[i]);
  }
  std::vector<T> q;
  std::vector<T> o_ref;
  std::vector<T> kv_data;
  std::vector<int32_t> kv_indptr{0};
  std::vector<int32_t> kv_indices;
  std::vector<int32_t> kv_last_page_len;
  size_t page_counter = 0;

  std::vector<std::vector<T>> keys, values;
  for (size_t i = 0; i < batch_size; ++i) {
    size_t seq_len = seq_lens[i];
    size_t num_pages = (seq_len + page_size - 1) / page_size;
    size_t last_page_len = (seq_len - 1) % page_size + 1;
    std::vector<T> qi(num_qo_heads * head_dim), ki(seq_len * num_kv_heads * head_dim),
        vi(seq_len * num_kv_heads * head_dim);
    utils::vec_normal_(qi);
    utils::vec_normal_(ki);
    utils::vec_normal_(vi);

    // compute reference output
    std::vector<T> o_ref_i =
        cpu_reference::single_mha<T, T>(qi, ki, vi, 1, seq_len, num_qo_heads, num_kv_heads,
                                        head_dim, false, QKVLayout::kNHD, rotary_mode);
    keys.push_back(ki);
    values.push_back(vi);
    // append new q and o_ref
    q.insert(q.end(), qi.begin(), qi.end());
    o_ref.insert(o_ref.end(), o_ref_i.begin(), o_ref_i.end());
    // append new kv_indptr, kv_indices and kv_last_page_len
    kv_last_page_len.push_back(last_page_len);
    kv_indptr.push_back(kv_indptr.back() + num_pages);
    for (size_t j = 0; j < num_pages; ++j) {
      kv_indices.push_back(page_counter++);
    }
  }
  kv_data.resize(page_counter * 2 * num_kv_heads * page_size * head_dim);
  utils::vec_zero_(kv_data);
  assert(q.size() == batch_size * num_qo_heads * head_dim);
  assert(o_ref.size() == batch_size * num_qo_heads * head_dim);

  flashinfer::paged_kv_t<PageStorage::kIndices, T, int32_t> paged_kv_cpu(
      num_kv_heads, page_size, head_dim, batch_size, kv_data.data(), kv_indices.data(),
      kv_indptr.data(), kv_last_page_len.data());
  cpu_reference::append_paged_kv_cache<T, int32_t>(paged_kv_cpu, keys, values, append_indptr);

  // copy data to device
  thrust::device_vector<T> kv_data_device(kv_data);
  thrust::device_vector<int32_t> kv_indptr_device(kv_indptr);
  thrust::device_vector<int32_t> kv_indices_device(kv_indices);
  thrust::device_vector<int32_t> kv_last_page_len_device(kv_last_page_len);
  thrust::device_vector<T> q_device(q);
  thrust::device_vector<T> o_device(o_ref.size());

  // create paged_kv object
  flashinfer::paged_kv_t<PageStorage::kIndices, T, int32_t> paged_kv(
      num_kv_heads, page_size, head_dim, batch_size,
      thrust::raw_pointer_cast(kv_data_device.data()),
      thrust::raw_pointer_cast(kv_indices_device.data()),
      thrust::raw_pointer_cast(kv_indptr_device.data()),
      thrust::raw_pointer_cast(kv_last_page_len_device.data()));
  flashinfer::BatchDecodeHandler<PageStorage::kIndices, T, T, int32_t> handler;
  handler.BeginForward(paged_kv, false, num_qo_heads, rotary_mode);

  if (!cooperative) {
    // use non-cooperative kernel
    cudaError_t status = flashinfer::BatchDecodeWithPagedKVCache<PageStorage::kIndices, T, T>(
        thrust::raw_pointer_cast(q_device.data()), paged_kv,
        thrust::raw_pointer_cast(o_device.data()), /*tmp=*/nullptr, /*lse=*/nullptr, num_qo_heads,
        rotary_mode);
    EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));
  } else {
    cudaError_t status =
        flashinfer::BatchDecodeWithPagedKVCacheWrapper<PageStorage::kIndices, T, T>(
            &handler, thrust::raw_pointer_cast(q_device.data()), paged_kv,
            thrust::raw_pointer_cast(o_device.data()), /*lse=*/nullptr, num_qo_heads, rotary_mode);
    EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));
  }
  // compare result
  thrust::host_vector<T> o_host = o_device;
  size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
  bool nan_detected = false;
  for (size_t i = 0; i < batch_size * num_qo_heads * head_dim; ++i) {
    if (std::isnan(float(o_host[i]))) {
      nan_detected = true;
    }
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(o_host[i]), float(o_ref[i]), 1e-3, 1e-3));
  }
  float result_accuracy = 1. - float(num_result_errors_atol_1e_3_rtol_1e_3) /
                                   float(batch_size * num_qo_heads * head_dim);
  std::cout << "page_size=" << page_size << ", num_qo_heads=" << num_qo_heads
            << ", num_kv_heads=" << num_kv_heads << ", batch_size=" << batch_size
            << ", head_dim=" << head_dim
            << ", rotary_mode=" << flashinfer::RotaryModeToString(rotary_mode)
            << ", result accuracy (atol=1e-3, rtol=1e-3): " << result_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.90) << "Result correctness test failed.";
  EXPECT_EQ(nan_detected, false) << "NaN detected.";
}

template <typename T>
void TestBatchDecodeKernelCorrectness() {
  for (size_t page_size : {1, 3, 7, 16}) {
    for (size_t batch_size : {1, 7, 37, 61}) {
      for (size_t num_qo_heads : {32}) {
        for (size_t num_kv_heads : {32, 4}) {
          for (size_t head_dim : {64, 128, 256}) {
            for (size_t rotary_mode : {0U, 1U}) {
              _TestBatchDecodingKernelCorrectness<T>(page_size, batch_size, num_qo_heads,
                                                     num_kv_heads, head_dim,
                                                     flashinfer::RotaryMode(rotary_mode), false);
            }
          }
        }
      }
    }
  }
}

template <typename T>
void TestCooperativeBatchDecodeKernelCorrectness() {
  for (size_t page_size : {1, 3, 7, 16}) {
    for (size_t batch_size : {1, 2, 4, 8}) {
      for (size_t num_qo_heads : {32}) {
        for (size_t num_kv_heads : {32, 4}) {
          for (size_t head_dim : {64, 128}) {
            for (size_t rotary_mode : {0U, 1U}) {
              _TestBatchDecodingKernelCorrectness<T>(page_size, batch_size, num_qo_heads,
                                                     num_kv_heads, head_dim,
                                                     flashinfer::RotaryMode(rotary_mode), true);
            }
          }
        }
      }
    }
  }
}

TEST(FlashInferCorrectnessTest, BatchDecodeKernelCorrectnessTestFP16) {
  TestBatchDecodeKernelCorrectness<half>();
}

TEST(FlashInferCorrectnessTest, TestBatchDecodeKernelCorrectnessBF16) {
  TestBatchDecodeKernelCorrectness<__nv_bfloat16>();
}

TEST(FlashInferCorrectnessTest, TestBatchDecodeKernelCorrectnessFP32) {
  TestBatchDecodeKernelCorrectness<float>();
}

TEST(FlashInferCorrectnessTest, TestBatchDecodeKernelCorrectnessE4M3) {
  TestBatchDecodeKernelCorrectness<__nv_fp8_e4m3>();
}

TEST(FlashInferCorrectnessTest, TestBatchDecodeKernelCorrectnessE5M2) {
  TestBatchDecodeKernelCorrectness<__nv_fp8_e5m2>();
}

TEST(FlashInferCorrectnessTest, TestCooperativeBatchDecodeKernelCorrectnessTestFP16) {
  TestCooperativeBatchDecodeKernelCorrectness<half>();
}
