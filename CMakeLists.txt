cmake_minimum_required(VERSION 3.25)

project(translation_models CXX C)
set(CMAKE_CXX_FLAGS "-fpermissive -fPIC -std=c++17")

add_executable(main main.cpp)

# Silero specific
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${TORCH_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

set(SILERO_DIR silero)
set(SILERO_SOURCES
      ${SILERO_DIR}/silero-vad-iter.h
      ${SILERO_DIR}/silero-vad-iter.cpp
      ${SILERO_DIR}/wav.h)
add_library(silero-vad SHARED ${SILERO_SOURCES})
target_link_libraries(silero-vad "${TORCH_LIBRARIES}")
message(STATUS "${TORCH_LIBRARIES}")

# Build dependencies
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/model)

# glib/GST dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_BASE REQUIRED gstreamer-base-1.0)

# Set headers
set(TVM_HOME model/3rdparty/tvm)

set(
  MODEL_INCLUDES
    ${TVM_HOME}/include
    ${TVM_HOME}/3rdparty/dlpack/include
    ${TVM_HOME}/3rdparty/dmlc-core/include
    ${TVM_HOME}/3rdparty/picojson
)

# libfex specific
set(
  LIBFEX_SRCS
    libs/chunker.h
    libs/chunker.cpp
    libs/callback-sink.h
    libs/callback-sink.cpp
    libs/fex.h
    libs/fex.cpp
    libs/logger.h
    libs/dr_wav.h
    libs/fex_original.h
)

add_library(fex SHARED ${LIBFEX_SRCS})

target_include_directories(
  fex
  PRIVATE
    ${MODEL_INCLUDES}
    ${GLIB_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
)

# Main app specific
target_include_directories(
  main
  PRIVATE
    ${MODEL_INCLUDES}
    ${GLIB_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
)

target_link_libraries(
  main
  PRIVATE
    silero-vad
    mlc_llm
    gstreamer-1.0
    gstbase-1.0
    gobject-2.0
    fex
)
