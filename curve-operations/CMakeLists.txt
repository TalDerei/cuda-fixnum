cmake_minimum_required(VERSION 3.10)

enable_language(CUDA)
string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -Wall,-Wextra --gpu-code=sm_60 --gpu-architecture=compute_60 --relocatable-device-code=true")

project (snark-challenge)

if("${DEPENDS_DIR}" STREQUAL "")
  set(
    DEPENDS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/depends"
    CACHE
    STRING
    "Optionally specify the dependency installation directory relative to the source directory (default: inside dependency folder)"
  )
else()
  set(DEPENDS_DIR "${DEPENDS_DIR}")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Common compilation flags and warning configuration
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -Wfatal-errors"
  )
  if("${MULTICORE}")
      if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
      else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
      endif()
  endif()
  # Default optimizations flags (to override, use -DOPT_FLAGS=...)
  if("${OPT_FLAGS}" STREQUAL "")
    set(
      OPT_FLAGS
      "-ggdb3 -O2 -march=westmere -mtune=skylake-avx512"
    )
  endif()
endif()

set(
  CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${OPT_FLAGS}"
)

find_path(GMP_INCLUDE_DIR NAMES gmp.h)
find_library(GMP_LIBRARIES NAMES gmp libgmp)
find_library(GMPXX_LIBRARIES NAMES gmpxx libgmpxx)

include(FindPkgConfig)
pkg_check_modules(
  CRYPTO
  REQUIRED

  libcrypto
)

if("${WITH_PROCPS}")
  pkg_check_modules(
    PROCPS
    REQUIRED

    libprocps
  )
else()
  add_definitions(
    -DNO_PROCPS
  )
endif()

# Enable Boost for program_options
FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

add_definitions(
  -DCURVE_${CURVE}
)

enable_testing()

include_directories(.)
include_directories(./src)
include_directories(./cuda-fixnum)

if("${DEBUG}")
  add_definitions(-DDEBUG=1)
endif()

if("${LOWMEM}")
  add_definitions(-DLOWMEM=1)
endif()

if("${MULTICORE}")
  add_definitions(-DMULTICORE=1)
endif()

if("${BINARY_OUTPUT}")
  add_definitions(-DBINARY_OUTPUT)
endif()

if("${MONTGOMERY_OUTPUT}")
  add_definitions(-DMONTGOMERY_OUTPUT)
endif()

if(NOT "${USE_PT_COMPRESSION}")
  add_definitions(-DNO_PT_COMPRESSION=1)
endif()

if("${PROFILE_OP_COUNTS}")
  add_definitions(-DPROFILE_OP_COUNTS=1)
endif()

if("${USE_MIXED_ADDITION}")
  add_definitions(-DUSE_MIXED_ADDITION=1)
endif()

if("${CPPDEBUG}")
  add_definitions(-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC)
endif()

if("${PERFORMANCE}")
  add_definitions(-DNDEBUG)
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -flto -fuse-linker-plugin"
  )
  set(
    CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -flto"
  )
endif()

if("${USE_ASM}")
  add_definitions(-DUSE_ASM)
endif()

find_program(
  MARKDOWN

  markdown_py
  DOC "Path to markdown_py binary"
)
if(MARKDOWN-NOTFOUND)
else()
   add_custom_target(
     doc
     ${MARKDOWN} -f ${CMAKE_CURRENT_BINARY_DIR}/README.html -x toc -x extra --noisy ${CMAKE_CURRENT_SOURCE_DIR}/README.md
     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
     COMMENT "Translating from markdown to HTML" VERBATIM
   )
endif()

# Configure CCache if available
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

# Add a `make check` target that builds and tests
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

add_subdirectory(depends)
add_subdirectory(libff)

add_executable (
    main
    EXCLUDE_FROM_ALL

    src/main.cu
    src/params.cu
    src/gpu_params.cu
    cuda-fixnum/compile.cu
    src/retrieve_utils.cu
    src/fq_mul.cu
    src/fq2_mul.cu
    src/fq3_mul.cu
    src/reduce_g1.cu
    src/reduce_mnt4_g2.cu
    src/reduce_mnt6_g2.cu
    src/io.cpp
    src/utils.cpp
    src/stage0.cpp
    src/stage1.cpp
    src/stage2.cpp
    src/stage3.cpp
)
target_link_libraries(main ff)
