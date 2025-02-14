# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/cuda-fixnum/curve-operations

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/cuda-fixnum/curve-operations/build

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/src/main.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/main.cu.o: ../src/main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/main.dir/src/main.cu.o"
	/usr/local/cuda-11.7/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/cuda-fixnum/curve-operations/src/main.cu -o CMakeFiles/main.dir/src/main.cu.o

CMakeFiles/main.dir/src/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/main.cu.o.requires:

.PHONY : CMakeFiles/main.dir/src/main.cu.o.requires

CMakeFiles/main.dir/src/main.cu.o.provides: CMakeFiles/main.dir/src/main.cu.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/main.cu.o.provides.build
.PHONY : CMakeFiles/main.dir/src/main.cu.o.provides

CMakeFiles/main.dir/src/main.cu.o.provides.build: CMakeFiles/main.dir/src/main.cu.o


CMakeFiles/main.dir/src/params.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/params.cu.o: ../src/params.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/main.dir/src/params.cu.o"
	/usr/local/cuda-11.7/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/cuda-fixnum/curve-operations/src/params.cu -o CMakeFiles/main.dir/src/params.cu.o

CMakeFiles/main.dir/src/params.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/params.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/params.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/params.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/params.cu.o.requires:

.PHONY : CMakeFiles/main.dir/src/params.cu.o.requires

CMakeFiles/main.dir/src/params.cu.o.provides: CMakeFiles/main.dir/src/params.cu.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/params.cu.o.provides.build
.PHONY : CMakeFiles/main.dir/src/params.cu.o.provides

CMakeFiles/main.dir/src/params.cu.o.provides.build: CMakeFiles/main.dir/src/params.cu.o


CMakeFiles/main.dir/src/gpu_params.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/gpu_params.cu.o: ../src/gpu_params.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/main.dir/src/gpu_params.cu.o"
	/usr/local/cuda-11.7/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/cuda-fixnum/curve-operations/src/gpu_params.cu -o CMakeFiles/main.dir/src/gpu_params.cu.o

CMakeFiles/main.dir/src/gpu_params.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/gpu_params.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/gpu_params.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/gpu_params.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/gpu_params.cu.o.requires:

.PHONY : CMakeFiles/main.dir/src/gpu_params.cu.o.requires

CMakeFiles/main.dir/src/gpu_params.cu.o.provides: CMakeFiles/main.dir/src/gpu_params.cu.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/gpu_params.cu.o.provides.build
.PHONY : CMakeFiles/main.dir/src/gpu_params.cu.o.provides

CMakeFiles/main.dir/src/gpu_params.cu.o.provides.build: CMakeFiles/main.dir/src/gpu_params.cu.o


CMakeFiles/main.dir/cuda-fixnum/compile.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/cuda-fixnum/compile.cu.o: ../cuda-fixnum/compile.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/main.dir/cuda-fixnum/compile.cu.o"
	/usr/local/cuda-11.7/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/cuda-fixnum/curve-operations/cuda-fixnum/compile.cu -o CMakeFiles/main.dir/cuda-fixnum/compile.cu.o

CMakeFiles/main.dir/cuda-fixnum/compile.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/main.dir/cuda-fixnum/compile.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/cuda-fixnum/compile.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/main.dir/cuda-fixnum/compile.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/cuda-fixnum/compile.cu.o.requires:

.PHONY : CMakeFiles/main.dir/cuda-fixnum/compile.cu.o.requires

CMakeFiles/main.dir/cuda-fixnum/compile.cu.o.provides: CMakeFiles/main.dir/cuda-fixnum/compile.cu.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/cuda-fixnum/compile.cu.o.provides.build
.PHONY : CMakeFiles/main.dir/cuda-fixnum/compile.cu.o.provides

CMakeFiles/main.dir/cuda-fixnum/compile.cu.o.provides.build: CMakeFiles/main.dir/cuda-fixnum/compile.cu.o


CMakeFiles/main.dir/src/retrieve_utils.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/retrieve_utils.cu.o: ../src/retrieve_utils.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object CMakeFiles/main.dir/src/retrieve_utils.cu.o"
	/usr/local/cuda-11.7/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/cuda-fixnum/curve-operations/src/retrieve_utils.cu -o CMakeFiles/main.dir/src/retrieve_utils.cu.o

CMakeFiles/main.dir/src/retrieve_utils.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/retrieve_utils.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/retrieve_utils.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/retrieve_utils.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/retrieve_utils.cu.o.requires:

.PHONY : CMakeFiles/main.dir/src/retrieve_utils.cu.o.requires

CMakeFiles/main.dir/src/retrieve_utils.cu.o.provides: CMakeFiles/main.dir/src/retrieve_utils.cu.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/retrieve_utils.cu.o.provides.build
.PHONY : CMakeFiles/main.dir/src/retrieve_utils.cu.o.provides

CMakeFiles/main.dir/src/retrieve_utils.cu.o.provides.build: CMakeFiles/main.dir/src/retrieve_utils.cu.o


CMakeFiles/main.dir/src/fq_mul.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/fq_mul.cu.o: ../src/fq_mul.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CUDA object CMakeFiles/main.dir/src/fq_mul.cu.o"
	/usr/local/cuda-11.7/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/cuda-fixnum/curve-operations/src/fq_mul.cu -o CMakeFiles/main.dir/src/fq_mul.cu.o

CMakeFiles/main.dir/src/fq_mul.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/fq_mul.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/fq_mul.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/fq_mul.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/fq_mul.cu.o.requires:

.PHONY : CMakeFiles/main.dir/src/fq_mul.cu.o.requires

CMakeFiles/main.dir/src/fq_mul.cu.o.provides: CMakeFiles/main.dir/src/fq_mul.cu.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/fq_mul.cu.o.provides.build
.PHONY : CMakeFiles/main.dir/src/fq_mul.cu.o.provides

CMakeFiles/main.dir/src/fq_mul.cu.o.provides.build: CMakeFiles/main.dir/src/fq_mul.cu.o


CMakeFiles/main.dir/src/fq2_mul.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/fq2_mul.cu.o: ../src/fq2_mul.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CUDA object CMakeFiles/main.dir/src/fq2_mul.cu.o"
	/usr/local/cuda-11.7/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/cuda-fixnum/curve-operations/src/fq2_mul.cu -o CMakeFiles/main.dir/src/fq2_mul.cu.o

CMakeFiles/main.dir/src/fq2_mul.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/fq2_mul.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/fq2_mul.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/fq2_mul.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/fq2_mul.cu.o.requires:

.PHONY : CMakeFiles/main.dir/src/fq2_mul.cu.o.requires

CMakeFiles/main.dir/src/fq2_mul.cu.o.provides: CMakeFiles/main.dir/src/fq2_mul.cu.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/fq2_mul.cu.o.provides.build
.PHONY : CMakeFiles/main.dir/src/fq2_mul.cu.o.provides

CMakeFiles/main.dir/src/fq2_mul.cu.o.provides.build: CMakeFiles/main.dir/src/fq2_mul.cu.o


CMakeFiles/main.dir/src/fq3_mul.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/fq3_mul.cu.o: ../src/fq3_mul.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CUDA object CMakeFiles/main.dir/src/fq3_mul.cu.o"
	/usr/local/cuda-11.7/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/cuda-fixnum/curve-operations/src/fq3_mul.cu -o CMakeFiles/main.dir/src/fq3_mul.cu.o

CMakeFiles/main.dir/src/fq3_mul.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/fq3_mul.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/fq3_mul.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/fq3_mul.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/fq3_mul.cu.o.requires:

.PHONY : CMakeFiles/main.dir/src/fq3_mul.cu.o.requires

CMakeFiles/main.dir/src/fq3_mul.cu.o.provides: CMakeFiles/main.dir/src/fq3_mul.cu.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/fq3_mul.cu.o.provides.build
.PHONY : CMakeFiles/main.dir/src/fq3_mul.cu.o.provides

CMakeFiles/main.dir/src/fq3_mul.cu.o.provides.build: CMakeFiles/main.dir/src/fq3_mul.cu.o


CMakeFiles/main.dir/src/reduce_g1.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/reduce_g1.cu.o: ../src/reduce_g1.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CUDA object CMakeFiles/main.dir/src/reduce_g1.cu.o"
	/usr/local/cuda-11.7/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/cuda-fixnum/curve-operations/src/reduce_g1.cu -o CMakeFiles/main.dir/src/reduce_g1.cu.o

CMakeFiles/main.dir/src/reduce_g1.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/reduce_g1.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/reduce_g1.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/reduce_g1.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/reduce_g1.cu.o.requires:

.PHONY : CMakeFiles/main.dir/src/reduce_g1.cu.o.requires

CMakeFiles/main.dir/src/reduce_g1.cu.o.provides: CMakeFiles/main.dir/src/reduce_g1.cu.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/reduce_g1.cu.o.provides.build
.PHONY : CMakeFiles/main.dir/src/reduce_g1.cu.o.provides

CMakeFiles/main.dir/src/reduce_g1.cu.o.provides.build: CMakeFiles/main.dir/src/reduce_g1.cu.o


CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o: ../src/reduce_mnt4_g2.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CUDA object CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o"
	/usr/local/cuda-11.7/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/cuda-fixnum/curve-operations/src/reduce_mnt4_g2.cu -o CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o

CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o.requires:

.PHONY : CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o.requires

CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o.provides: CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o.provides.build
.PHONY : CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o.provides

CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o.provides.build: CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o


CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o: ../src/reduce_mnt6_g2.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CUDA object CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o"
	/usr/local/cuda-11.7/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/cuda-fixnum/curve-operations/src/reduce_mnt6_g2.cu -o CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o

CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o.requires:

.PHONY : CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o.requires

CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o.provides: CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o.provides.build
.PHONY : CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o.provides

CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o.provides.build: CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o


CMakeFiles/main.dir/src/io.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/io.cpp.o: ../src/io.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/main.dir/src/io.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/io.cpp.o -c /home/ubuntu/cuda-fixnum/curve-operations/src/io.cpp

CMakeFiles/main.dir/src/io.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/io.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/cuda-fixnum/curve-operations/src/io.cpp > CMakeFiles/main.dir/src/io.cpp.i

CMakeFiles/main.dir/src/io.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/io.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/cuda-fixnum/curve-operations/src/io.cpp -o CMakeFiles/main.dir/src/io.cpp.s

CMakeFiles/main.dir/src/io.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/src/io.cpp.o.requires

CMakeFiles/main.dir/src/io.cpp.o.provides: CMakeFiles/main.dir/src/io.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/io.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/io.cpp.o.provides

CMakeFiles/main.dir/src/io.cpp.o.provides.build: CMakeFiles/main.dir/src/io.cpp.o


CMakeFiles/main.dir/src/utils.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/utils.cpp.o: ../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object CMakeFiles/main.dir/src/utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/utils.cpp.o -c /home/ubuntu/cuda-fixnum/curve-operations/src/utils.cpp

CMakeFiles/main.dir/src/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/cuda-fixnum/curve-operations/src/utils.cpp > CMakeFiles/main.dir/src/utils.cpp.i

CMakeFiles/main.dir/src/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/cuda-fixnum/curve-operations/src/utils.cpp -o CMakeFiles/main.dir/src/utils.cpp.s

CMakeFiles/main.dir/src/utils.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/src/utils.cpp.o.requires

CMakeFiles/main.dir/src/utils.cpp.o.provides: CMakeFiles/main.dir/src/utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/utils.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/utils.cpp.o.provides

CMakeFiles/main.dir/src/utils.cpp.o.provides.build: CMakeFiles/main.dir/src/utils.cpp.o


CMakeFiles/main.dir/src/stage0.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/stage0.cpp.o: ../src/stage0.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object CMakeFiles/main.dir/src/stage0.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/stage0.cpp.o -c /home/ubuntu/cuda-fixnum/curve-operations/src/stage0.cpp

CMakeFiles/main.dir/src/stage0.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/stage0.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/cuda-fixnum/curve-operations/src/stage0.cpp > CMakeFiles/main.dir/src/stage0.cpp.i

CMakeFiles/main.dir/src/stage0.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/stage0.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/cuda-fixnum/curve-operations/src/stage0.cpp -o CMakeFiles/main.dir/src/stage0.cpp.s

CMakeFiles/main.dir/src/stage0.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/src/stage0.cpp.o.requires

CMakeFiles/main.dir/src/stage0.cpp.o.provides: CMakeFiles/main.dir/src/stage0.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/stage0.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/stage0.cpp.o.provides

CMakeFiles/main.dir/src/stage0.cpp.o.provides.build: CMakeFiles/main.dir/src/stage0.cpp.o


CMakeFiles/main.dir/src/stage1.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/stage1.cpp.o: ../src/stage1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object CMakeFiles/main.dir/src/stage1.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/stage1.cpp.o -c /home/ubuntu/cuda-fixnum/curve-operations/src/stage1.cpp

CMakeFiles/main.dir/src/stage1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/stage1.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/cuda-fixnum/curve-operations/src/stage1.cpp > CMakeFiles/main.dir/src/stage1.cpp.i

CMakeFiles/main.dir/src/stage1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/stage1.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/cuda-fixnum/curve-operations/src/stage1.cpp -o CMakeFiles/main.dir/src/stage1.cpp.s

CMakeFiles/main.dir/src/stage1.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/src/stage1.cpp.o.requires

CMakeFiles/main.dir/src/stage1.cpp.o.provides: CMakeFiles/main.dir/src/stage1.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/stage1.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/stage1.cpp.o.provides

CMakeFiles/main.dir/src/stage1.cpp.o.provides.build: CMakeFiles/main.dir/src/stage1.cpp.o


CMakeFiles/main.dir/src/stage2.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/stage2.cpp.o: ../src/stage2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object CMakeFiles/main.dir/src/stage2.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/stage2.cpp.o -c /home/ubuntu/cuda-fixnum/curve-operations/src/stage2.cpp

CMakeFiles/main.dir/src/stage2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/stage2.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/cuda-fixnum/curve-operations/src/stage2.cpp > CMakeFiles/main.dir/src/stage2.cpp.i

CMakeFiles/main.dir/src/stage2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/stage2.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/cuda-fixnum/curve-operations/src/stage2.cpp -o CMakeFiles/main.dir/src/stage2.cpp.s

CMakeFiles/main.dir/src/stage2.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/src/stage2.cpp.o.requires

CMakeFiles/main.dir/src/stage2.cpp.o.provides: CMakeFiles/main.dir/src/stage2.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/stage2.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/stage2.cpp.o.provides

CMakeFiles/main.dir/src/stage2.cpp.o.provides.build: CMakeFiles/main.dir/src/stage2.cpp.o


CMakeFiles/main.dir/src/stage3.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/stage3.cpp.o: ../src/stage3.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building CXX object CMakeFiles/main.dir/src/stage3.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/src/stage3.cpp.o -c /home/ubuntu/cuda-fixnum/curve-operations/src/stage3.cpp

CMakeFiles/main.dir/src/stage3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/stage3.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/cuda-fixnum/curve-operations/src/stage3.cpp > CMakeFiles/main.dir/src/stage3.cpp.i

CMakeFiles/main.dir/src/stage3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/stage3.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/cuda-fixnum/curve-operations/src/stage3.cpp -o CMakeFiles/main.dir/src/stage3.cpp.s

CMakeFiles/main.dir/src/stage3.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/src/stage3.cpp.o.requires

CMakeFiles/main.dir/src/stage3.cpp.o.provides: CMakeFiles/main.dir/src/stage3.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/stage3.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/src/stage3.cpp.o.provides

CMakeFiles/main.dir/src/stage3.cpp.o.provides.build: CMakeFiles/main.dir/src/stage3.cpp.o


# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/src/main.cu.o" \
"CMakeFiles/main.dir/src/params.cu.o" \
"CMakeFiles/main.dir/src/gpu_params.cu.o" \
"CMakeFiles/main.dir/cuda-fixnum/compile.cu.o" \
"CMakeFiles/main.dir/src/retrieve_utils.cu.o" \
"CMakeFiles/main.dir/src/fq_mul.cu.o" \
"CMakeFiles/main.dir/src/fq2_mul.cu.o" \
"CMakeFiles/main.dir/src/fq3_mul.cu.o" \
"CMakeFiles/main.dir/src/reduce_g1.cu.o" \
"CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o" \
"CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o" \
"CMakeFiles/main.dir/src/io.cpp.o" \
"CMakeFiles/main.dir/src/utils.cpp.o" \
"CMakeFiles/main.dir/src/stage0.cpp.o" \
"CMakeFiles/main.dir/src/stage1.cpp.o" \
"CMakeFiles/main.dir/src/stage2.cpp.o" \
"CMakeFiles/main.dir/src/stage3.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/main.cu.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/params.cu.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/gpu_params.cu.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/cuda-fixnum/compile.cu.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/retrieve_utils.cu.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/fq_mul.cu.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/fq2_mul.cu.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/fq3_mul.cu.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/reduce_g1.cu.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/io.cpp.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/utils.cpp.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/stage0.cpp.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/stage1.cpp.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/stage2.cpp.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/src/stage3.cpp.o
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/build.make
CMakeFiles/main.dir/cmake_device_link.o: libff/libff.a
CMakeFiles/main.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libgmp.so
CMakeFiles/main.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libgmpxx.so
CMakeFiles/main.dir/cmake_device_link.o: CMakeFiles/main.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Linking CUDA device code CMakeFiles/main.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: CMakeFiles/main.dir/cmake_device_link.o

.PHONY : CMakeFiles/main.dir/build

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/src/main.cu.o" \
"CMakeFiles/main.dir/src/params.cu.o" \
"CMakeFiles/main.dir/src/gpu_params.cu.o" \
"CMakeFiles/main.dir/cuda-fixnum/compile.cu.o" \
"CMakeFiles/main.dir/src/retrieve_utils.cu.o" \
"CMakeFiles/main.dir/src/fq_mul.cu.o" \
"CMakeFiles/main.dir/src/fq2_mul.cu.o" \
"CMakeFiles/main.dir/src/fq3_mul.cu.o" \
"CMakeFiles/main.dir/src/reduce_g1.cu.o" \
"CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o" \
"CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o" \
"CMakeFiles/main.dir/src/io.cpp.o" \
"CMakeFiles/main.dir/src/utils.cpp.o" \
"CMakeFiles/main.dir/src/stage0.cpp.o" \
"CMakeFiles/main.dir/src/stage1.cpp.o" \
"CMakeFiles/main.dir/src/stage2.cpp.o" \
"CMakeFiles/main.dir/src/stage3.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/src/main.cu.o
main: CMakeFiles/main.dir/src/params.cu.o
main: CMakeFiles/main.dir/src/gpu_params.cu.o
main: CMakeFiles/main.dir/cuda-fixnum/compile.cu.o
main: CMakeFiles/main.dir/src/retrieve_utils.cu.o
main: CMakeFiles/main.dir/src/fq_mul.cu.o
main: CMakeFiles/main.dir/src/fq2_mul.cu.o
main: CMakeFiles/main.dir/src/fq3_mul.cu.o
main: CMakeFiles/main.dir/src/reduce_g1.cu.o
main: CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o
main: CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o
main: CMakeFiles/main.dir/src/io.cpp.o
main: CMakeFiles/main.dir/src/utils.cpp.o
main: CMakeFiles/main.dir/src/stage0.cpp.o
main: CMakeFiles/main.dir/src/stage1.cpp.o
main: CMakeFiles/main.dir/src/stage2.cpp.o
main: CMakeFiles/main.dir/src/stage3.cpp.o
main: CMakeFiles/main.dir/build.make
main: libff/libff.a
main: /usr/lib/x86_64-linux-gnu/libgmp.so
main: /usr/lib/x86_64-linux-gnu/libgmpxx.so
main: CMakeFiles/main.dir/cmake_device_link.o
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_19) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main

.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/main.cu.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/params.cu.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/gpu_params.cu.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/cuda-fixnum/compile.cu.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/retrieve_utils.cu.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/fq_mul.cu.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/fq2_mul.cu.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/fq3_mul.cu.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/reduce_g1.cu.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/reduce_mnt4_g2.cu.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/reduce_mnt6_g2.cu.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/io.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/utils.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/stage0.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/stage1.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/stage2.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/src/stage3.cpp.o.requires

.PHONY : CMakeFiles/main.dir/requires

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /home/ubuntu/cuda-fixnum/curve-operations/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/cuda-fixnum/curve-operations /home/ubuntu/cuda-fixnum/curve-operations /home/ubuntu/cuda-fixnum/curve-operations/build /home/ubuntu/cuda-fixnum/curve-operations/build /home/ubuntu/cuda-fixnum/curve-operations/build/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

