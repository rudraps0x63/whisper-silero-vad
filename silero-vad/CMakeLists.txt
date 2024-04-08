option(ENABLE_DEBUG_SPEECH_PROB "Enable debug speech probability output" OFF)

cmake_minimum_required(VERSION 3.25 FATAL_ERROR) # Use the higher version
project(silero-vad)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Set the source file and executable name
set(
  SOURCES
    src/silero-vad-iter.h
    src/silero-vad-iter.cpp
    src/wav.h
)

# Add the executable using the variables without quotes
add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# The following code block is optional and only for Windows.
# It copies the DLLs needed for LibTorch to the directory of the executable to avoid runtime errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${EXECUTABLE_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${EXECUTABLE_NAME}>)
endif (MSVC)

if(ENABLE_DEBUG_SPEECH_PROB)
  target_compile_definitions(${EXECUTABLE_NAME} PRIVATE __DEBUG_SPEECH_PROB___)
endif()
