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
#include "flashinfer_ops.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("single_decode_with_kv_cache", &single_decode_with_kv_cache,
        "Single-request decode with KV-Cache operator");
  m.def("single_prefill_with_kv_cache", &single_prefill_with_kv_cache,
        "Single-request prefill with KV-Cache operator");
  m.def("single_prefill_with_kv_cache_return_lse", &single_prefill_with_kv_cache_return_lse,
        "Single-request prefill with KV-Cache operator, return logsumexp");
  m.def("merge_state", &merge_state, "Merge two self-attention states");
  m.def("merge_states", &merge_states, "Merge multiple self-attention states");
  m.def("batch_decode_with_padded_kv_cache", &batch_decode_with_padded_kv_cache,
        "Multi-request batch decode with padded KV-Cache operator");
  m.def("batch_decode_with_padded_kv_cache_return_lse",
        &batch_decode_with_padded_kv_cache_return_lse,
        "Multi-request batch decode with padded KV-Cache operator, return logsumexp");
}
