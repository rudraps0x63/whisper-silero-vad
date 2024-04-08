
/*!
 *  Copyright (c) 2023 by Contributors
 * \file translate.cc
 * \brief Implementation of translation.
 */
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>

#include "base.h"

namespace mlc {
namespace llm {

MLC_LLM_DLL tvm::runtime::Module CreateTranslateModule(DLDevice device);

}  // namespace llm
}  // namespace mlc
