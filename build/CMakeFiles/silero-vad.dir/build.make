# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rudra/dev/work/PearTree/AI/whisper-plus-drwav-2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rudra/dev/work/PearTree/AI/whisper-plus-drwav-2/build

# Include any dependencies generated for this target.
include CMakeFiles/silero-vad.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/silero-vad.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/silero-vad.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/silero-vad.dir/flags.make

CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.o: CMakeFiles/silero-vad.dir/flags.make
CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.o: /home/rudra/dev/work/PearTree/AI/whisper-plus-drwav-2/silero/silero-vad-iter.cpp
CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.o: CMakeFiles/silero-vad.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/rudra/dev/work/PearTree/AI/whisper-plus-drwav-2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.o -MF CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.o.d -o CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.o -c /home/rudra/dev/work/PearTree/AI/whisper-plus-drwav-2/silero/silero-vad-iter.cpp

CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rudra/dev/work/PearTree/AI/whisper-plus-drwav-2/silero/silero-vad-iter.cpp > CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.i

CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rudra/dev/work/PearTree/AI/whisper-plus-drwav-2/silero/silero-vad-iter.cpp -o CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.s

# Object files for target silero-vad
silero__vad_OBJECTS = \
"CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.o"

# External object files for target silero-vad
silero__vad_EXTERNAL_OBJECTS =

libsilero-vad.so: CMakeFiles/silero-vad.dir/silero/silero-vad-iter.cpp.o
libsilero-vad.so: CMakeFiles/silero-vad.dir/build.make
libsilero-vad.so: /home/rudra/dev/work/PearTree/AI/libtorch/lib/libtorch.so
libsilero-vad.so: /home/rudra/dev/work/PearTree/AI/libtorch/lib/libc10.so
libsilero-vad.so: /home/rudra/dev/work/PearTree/AI/libtorch/lib/libkineto.a
libsilero-vad.so: /home/rudra/dev/work/PearTree/AI/libtorch/lib/libc10.so
libsilero-vad.so: CMakeFiles/silero-vad.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/rudra/dev/work/PearTree/AI/whisper-plus-drwav-2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libsilero-vad.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/silero-vad.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/silero-vad.dir/build: libsilero-vad.so
.PHONY : CMakeFiles/silero-vad.dir/build

CMakeFiles/silero-vad.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/silero-vad.dir/cmake_clean.cmake
.PHONY : CMakeFiles/silero-vad.dir/clean

CMakeFiles/silero-vad.dir/depend:
	cd /home/rudra/dev/work/PearTree/AI/whisper-plus-drwav-2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rudra/dev/work/PearTree/AI/whisper-plus-drwav-2 /home/rudra/dev/work/PearTree/AI/whisper-plus-drwav-2 /home/rudra/dev/work/PearTree/AI/whisper-plus-drwav-2/build /home/rudra/dev/work/PearTree/AI/whisper-plus-drwav-2/build /home/rudra/dev/work/PearTree/AI/whisper-plus-drwav-2/build/CMakeFiles/silero-vad.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/silero-vad.dir/depend

